import xml.etree.ElementTree as ET
from collections.abc import Callable
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from json2xml.json2xml import Json2xml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

from ._version import __version__, __version_tuple__
from .core import Plasma, PsiGrid
from .input import MachineIn
from .post_process import Machine
from .util import is_polygon_closed


class DS(ABC):
    def __getitem__(self, key: str | int) -> Any:
        return self._getitem(self._separate_key(key))

    def __setitem__(self, key: str | int, value: Any) -> None:
        self._setitem(self._separate_key(key), value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and hasattr(super(), "__getattr__"):
            return super().__getattr__(name)  # type: ignore [misc]
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def _separate_key(self, key: str | int) -> list[str | int]:
        if isinstance(key, int):
            return [key]
        split_key = key.replace("/", ".").replace("]", "").replace("[", ".").split(".")
        return [int(k) if k.isdecimal() else k for k in split_key]

    @abstractmethod
    def _wrap_object(self, o: Any, force: bool) -> Any: ...

    @abstractmethod
    def _getitem(self, key: list[str | int]) -> Any: ...

    @abstractmethod
    def _setitem(self, key: list[str | int], value: Any) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def inner(self) -> Any: ...


def add_imas_code_info(ds: DS, input_data: MachineIn | None = None) -> None:
    """Add the code info to an OMAS data structure"""
    code = ds["code"]
    code["name"] = "DipolEQ"
    code["description"] = "Dipole equilibrium solver"
    code["repository"] = "https://github.com/dgarnier/dipoleq"
    # return tag of release or commit hash depending on if its a release or not
    code["version"] = (
        f"v{__version__}"
        if len(__version_tuple__) < 5
        else str(__version_tuple__[4]).split(".", maxsplit=1)[0]
    )
    # could add the input data as XML, but... ugh
    # needs pydantic_xml, xmltodict doesn't preserve types going roundtrip.
    # but pydantic works.. so lets just save as json.
    if input_data:
        input_dict = input_data.model_dump(mode="json", exclude_unset=True)
        xml = Json2xml(input_dict, wrapper="Machine", pretty=True).to_xml()
        code["parameters"] = xml

    # input_data.model_dump(mode=)
    # could add which libraries and commits to add, but, really
    # this is enough to reproduce the results
    # one would encode the input data as XML into a string
    # code['output_flag'][eq_time_index] = 0 if run is successful,
    # negative if not to be used


def mas_input_params(equilibrium: DS) -> dict[str, Any] | None:
    """Create a DipolEQ input data from an equilibrium data structure"""
    # need to create the MachineIn object from the input data
    # which is stored in the code section of the data structure.
    # this is a bit of a pain, but it is what it is.
    # the input data is stored as XML in the code section
    # so we need to extract it and then load it into the MachineIn object.
    if equilibrium["code.name"] != "DipolEQ":
        return None

    def xml_to_dict(node: ET.Element) -> Any:
        """Compatible conversion from XML based on Json2xml output"""
        match node.attrib.get("type", "dict"):
            case "dict":
                return {child.tag: xml_to_dict(child) for child in node}
            case "list":
                return [xml_to_dict(child) for child in node]
            case "str":
                return node.text
            case "int":
                return int(node.text) if node.text is not None else None
            case "float":
                return float(node.text) if node.text is not None else None
            case "bool":
                return node.text and node.text.lower() in ("true", "yes", "t", "1")

    data = xml_to_dict(ET.fromstring(equilibrium["code.parameters"]))
    return data if isinstance(data, dict) else None


def add_limiters(m: Machine, wall: DS) -> None:
    """Add the limiter as a IMAS/OMAS wall structure
    For limiters, there are different types.
    Each type can have a name, index, and description.
    index = 0 = single contour
            1 = PFC structure block
    the limiter is also made of units.. which can
    also have component types.
    """
    # add data providence to wall structure
    add_imas_code_info(wall)

    # IMAS units should be contiguous and clockwise
    # of course, the inner limiter will be done in reverse
    # since it limits on the outside of its surface.
    wall0 = wall["description_2d[0]"]
    wall0["type.name"] = "Limiter(s)"
    wall0["type.index"] = 1  # multiple limiter units, no vessel
    wall0["type.description"] = (
        "As a dipole, this will contain outer and inner limiters"
    )
    wall0["limiter.type.index"] = 0  # official single contour limiter
    wall0["limiter.type.name"] = "Limiter Contour(s)"

    # outer limiter first
    outline = m.Limiters.olim_outline()
    unit = wall0["limiter.unit[0]"]
    unit["name"] = "Outer limiter"
    unit["identifier"] = "outer_limiter"
    if is_polygon_closed(outline):  # determine closed by last
        # unit['outline.closed'] = 1  .. not in IMAS schema
        outline = outline[:-1]  # remove last point for closed
    # else:
    #    unit['outline.closed'] = 0
    unit["outline.r"] = outline[:, 0]
    unit["outline.z"] = outline[:, 1]
    unit["component_type.name"] = "outer_limiter"
    unit["component_type.index"] = 5  # 5 = limiter

    # inner limiter
    outline = m.Limiters.ilim_outline()
    unit = wall0["limiter.unit[1]"]
    unit["name"] = "inner_limiter"
    unit["identifier"] = "inner_limiter"
    if is_polygon_closed(outline):  # determine closed by last
        #    unit['outline.closed'] = 1
        outline = outline[:-1]
    # else:
    #    unit['outline.closed'] = 0
    unit["outline.r"] = outline[:, 0]
    unit["outline.z"] = outline[:, 1]
    unit["component_type.name"] = "inner_limiter"
    unit["component_type.index"] = -5  # 5 = limiter, but "private/custom" type


def add_boundary(m: Machine, ts: DS) -> None:
    """Add the boundary data to an IMAS/OMAS equilibrium time slice"""
    # the values here are taken from the IMAS/OMAS schema
    # https://imas-data-dictionary.readthedocs.io/en/latest/generated/ids/equilibrium.html
    # https://gafusion.github.io/omas/schema/schema_equilibrium.html
    # make boundary not quite at the separatrix
    # this is the 99.5% flux surface
    pg = m.PsiGrid
    psi_norm = m.Plasma.PsiXmax  # set by input
    bound = ts["boundary"]
    bnd_r, bnd_z = pg.get_contour(psi_norm)
    bound["outline.r"] = bnd_r
    bound["outline.z"] = bnd_z
    bound["psi_norm"] = psi_norm
    bound["psi"] = (pg.PsiLim - pg.PsiAxis) * psi_norm + pg.PsiAxis
    bound["minor_radius"] = (np.max(bnd_r) - np.min(bnd_r)) / 2
    bound["type"] = 1 if m.is_diverted() else 0  # type: ignore[attr-defined]


def add_boundary_separatrix(m: Machine, ts: DS) -> None:
    """Add the boundary separatrix data to an IMAS/OMAS equilibrium time slice"""
    # the values here are taken from the IMAS/OMAS schema
    # https://imas-data-dictionary.readthedocs.io/en/latest/generated/ids/equilibrium.html
    # https://gafusion.github.io/omas/schema/schema_equilibrium.html
    sep_r, sep_z = m.PsiGrid.get_contour(1.0)
    bsep = ts["boundary_separatrix"]
    bsep["outline.r"] = sep_r
    bsep["outline.z"] = sep_z
    bsep["psi"] = m.PsiGrid.PsiLim
    if m.is_diverted():  # type: ignore[attr-defined]
        bsep["type"] = 1
    else:
        bsep["type"] = 0
        active_point = m.get_outer_limiter_contact_point()  # type: ignore[attr-defined]
        if active_point is not None:
            bsep["active_limiter_point.r"] = active_point[0]
            bsep["active_limiter_point.z"] = active_point[1]
    # not sure how useful this is if not diverted, but
    # decided to add them anyway.. would be more helpful
    # with the flux values there.
    for i, xp in enumerate(m.get_x_points()):  # type: ignore[attr-defined]
        bsep[f"x_point[{i}].r"] = xp.Rs
        bsep[f"x_point[{i}].z"] = xp.Zs
        # this doesn't exist in IMAS schema
        # bsep[f'x_point[{i}].psi'] = xp.Psi

    # FIXME:  other things to calculate..
    # closest_wall_point
    # dr_dz_zero_point, geometric_axis
    # elongation, elongation_upper, elongation_lower
    # triangularity, triangularity_upper, triangularity_lower
    # triangularity_inner, triangularity_outer
    # squareness (in alpha)
    # strikepoint(s), gap(s)


def add_inner_boundary_separatrix(m: Machine, ts: DS) -> None:
    """Add the inner boundary separatrix data to an equilibrium time slice"""
    if m.PsiGrid.PsiAxis == m.PsiGrid.PsiMagAxis:
        # no fcfs when the inner boundary is at the axis.
        return
    sep_r, sep_z = m.PsiGrid.get_contour(0.0)
    bsep = ts["inner_boundary_separatrix"]
    bsep["outline.r"] = sep_r
    bsep["outline.z"] = sep_z
    bsep["psi"] = m.PsiGrid.PsiAxis
    bsep["type"] = 0
    active_point = m.get_inner_limiter_contact_point()  # type: ignore[attr-defined]
    if active_point is not None:
        bsep["active_limiter_point.r"] = active_point[0]
        bsep["active_limiter_point.z"] = active_point[1]


def fill_ds(m: Machine, eq: DS, wall: DS, time_index: int | None, time: float) -> None:
    """Add all the equilibrim and wall information from a DipolEQ Machine
    into the equilibrium and wall data structures.
    """
    pl = m.Plasma
    pg = m.PsiGrid

    flux_surface_averager = FluxSurfaceAverager(m)
    r_rmax, z_rmax = flux_surface_averager.rmax()
    r_zmax, z_zmax = flux_surface_averager.zmax()
    r_rmin, z_rmin = flux_surface_averager.rmin()
    r_zmin, z_zmin = flux_surface_averager.zmin()

    add_limiters(m, wall)

    # add the structure and the wall from the machine
    input_data = getattr(m, "input_data", None)
    add_imas_code_info(eq, input_data=input_data)

    if time_index is None:
        time_index = len(eq["time_slice"])
    eqt = eq["time_slice"][time_index]

    # the values here are taken from the IMAS/OMAS schema
    # https://imas-data-dictionary.readthedocs.io/en/latest/generated/ids/equilibrium.html
    # https://gafusion.github.io/omas/schema/schema_equilibrium.html

    psi = np.asarray(pl.Psi_pr)  # flux values, 1d
    psi_norm = np.asarray(pl.PsiX_pr)

    # Minor radius, defined as half the difference of minimum and maximum radius of the flux surface
    a = (r_rmax - r_rmin) / 2
    # Toroidal field function
    f = np.asarray(pl.G_pr) * pl.B0R0

    R_grid = np.stack([np.asarray(pg.R)] * len(np.asarray(pg.Z)), axis=0)
    B2_grid = np.asarray(pl.B2)
    grad_psi2_grid = np.asarray(pl.GradPsi2)
    avg_1_over_R = flux_surface_averager.flux_surface_average(1 / R_grid)
    avg_1_over_R2 = flux_surface_averager.flux_surface_average(1 / R_grid**2)
    avg_grad_psi = flux_surface_averager.flux_surface_average(np.sqrt(grad_psi2_grid))
    avg_grad_psi2 = flux_surface_averager.flux_surface_average(grad_psi2_grid)
    avg_grad_psi2_over_R2 = flux_surface_averager.flux_surface_average(
        grad_psi2_grid / R_grid**2
    )
    avg_grad_psi2_over_B2 = flux_surface_averager.flux_surface_average(
        grad_psi2_grid / B2_grid
    )

    # Phi is defined as a double integral of the toroidal magnetic field over the cross-sectional surface contained within the flux contour.
    dphi_dvol = f / (2 * np.pi) * avg_1_over_R2
    # For tokamaks, the magnetic axis forms the start of the integral, so the initial Phi is zero.
    # However, dipoles have a finite FCFS, and it turns out we need to choose a non-zero initial Phi.
    # The actual value though is fairly unimportant, so we approximate the contained toroidal magnetic flux.
    phi_0 = np.pi * pl.B0 * a[0] ** 2
    phi = cumulative_trapezoid(dphi_dvol, x=np.asarray(pl.Vol_pr), initial=0) + phi_0
    rho_tor = np.sqrt(phi / (np.pi * pl.B0))
    drho_dphi = 1 / (2 * np.pi * pl.B0 * rho_tor)
    dvol_dpsi = np.asarray(pl.Volp_pr)

    drho_dpsi = drho_dphi * dphi_dvol * dvol_dpsi

    avg_grad_rho = drho_dpsi * avg_grad_psi
    avg_grad_rho2 = drho_dpsi**2 * avg_grad_psi2
    avg_grad_rho2_over_B2 = drho_dpsi**2 * avg_grad_psi2_over_B2
    avg_grad_rho2_over_R2 = drho_dpsi**2 * avg_grad_psi2_over_R2

    MU0 = 4.0e-7 * np.pi
    j_grid = np.asarray(m.PsiGrid.Current) / MU0

    # Set the time array
    eqt["time"] = time
    eq[f"time.{time_index}"] = time

    # 0D quantities
    glob = eqt["global_quantities"]
    glob["psi_axis"] = pg.PsiMagAxis
    glob["psi_boundary"] = pg.PsiLim
    glob["psi_inner_boundary"] = pg.PsiAxis
    glob["magnetic_axis.r"] = pg.RMagAxis
    glob["magnetic_axis.z"] = pg.ZMagAxis
    glob["ip"] = pl.Ip

    # B0, R0 is weird
    eq["vacuum_toroidal_field.r0"] = pl.R0
    eq[f"vacuum_toroidal_field.b0.{time_index}"] = pl.B0

    # 1D quantities
    eq1d = eqt["profiles_1d"]
    eq1d["psi"] = psi
    eq1d["psi_norm"] = psi_norm
    eq1d["phi"] = phi
    eq1d["pressure"] = np.asarray(pl.P_pr)
    eq1d["f"] = f
    eq1d["dpressure_dpsi"] = np.asarray(pl.Pp_pr)
    eq1d["f_df_dpsi"] = np.asarray(pl.G2p_pr) * (pl.B0R0) ** 2
    eq1d["j_tor"] = (
        flux_surface_averager.flux_surface_average(j_grid / R_grid) / avg_1_over_R
    )
    eq1d["q"] = np.asarray(pl.q_pr)
    eq1d["r_inboard"] = r_rmin
    eq1d["r_outboard"] = r_rmax
    eq1d["rho_tor"] = rho_tor
    eq1d["rho_tor_norm"] = (rho_tor - rho_tor[0]) / (rho_tor[-1] - rho_tor[0])
    eq1d["elongation"] = (r_zmax - r_zmin) / (r_rmax - r_rmin)
    eq1d["triangularity_upper"] = (pl.R0 - r_zmax) / a
    eq1d["triangularity_lower"] = (pl.R0 - r_zmin) / a
    eq1d["volume"] = np.asarray(pl.Vol_pr)
    eq1d["dvolume_dpsi"] = dvol_dpsi
    eq1d["dvolume_drho_tor"] = dvol_dpsi / drho_dpsi
    eq1d["gm1"] = avg_1_over_R2
    eq1d["gm2"] = avg_grad_rho2_over_R2
    eq1d["gm3"] = avg_grad_rho2
    eq1d["gm4"] = flux_surface_averager.flux_surface_average(1 / B2_grid)
    eq1d["gm5"] = flux_surface_averager.flux_surface_average(B2_grid)
    eq1d["gm6"] = avg_grad_rho2_over_B2
    eq1d["gm7"] = avg_grad_rho
    eq1d["gm8"] = flux_surface_averager.flux_surface_average(R_grid)
    eq1d["gm9"] = avg_1_over_R

    # 2D quantities
    eq2d = eqt["profiles_2d.0"]
    eq2d["type.index"] = 0  # total fields.. could also be broken down into components
    eq2d["grid_type.index"] = 1  # regular R,Z grid
    eq2d["grid_type.name"] = "RZ"
    eq2d["grid.dim1"] = np.asarray(m.PsiGrid.R)
    eq2d["grid.dim2"] = np.asarray(m.PsiGrid.Z)
    eq2d["psi"] = np.asarray(m.PsiGrid.Psi)
    eq2d["j_tor"] = j_grid
    eq2d["b_field_r"] = np.asarray(pl.GradPsiZ) / (2 * np.pi * R_grid)
    eq2d["b_field_z"] = -np.asarray(pl.GradPsiR) / (2 * np.pi * R_grid)
    eq2d["b_field_tor"] = np.asarray(pl.Bt)
    # others to add
    # eq2d['grid.volume_element']

    # boundaries
    add_boundary(m, eqt)
    add_boundary_separatrix(m, eqt)
    add_inner_boundary_separatrix(m, eqt)


class FluxSurfaceAverager:
    """
    Class to handle flux surface averaging of quantities defined on the PsiGrid.
    It can also calculate the extrema of each flux surface, giving the (R, Z) coordinates of the four extrema (min R, max R, min Z, max Z).
    """

    m: Machine
    contours: list[
        tuple[float, NDArray[np.float64], NDArray[np.float64]]
    ]  # list of (r, z) arrays for each flux surface
    arc_length_coords: list[NDArray[np.float64]]
    B_p: list[NDArray[np.float64]]

    def __init__(self, m: Machine) -> None:
        self.m = m
        self.make_contours()

    def make_contours(self) -> None:
        self.contours = [
            (psi_n, *self.pg.get_contour(psi_n))
            for psi_n in np.asarray(self.pl.PsiX_pr)
        ]

        self.B_p = self.grid_to_flux(np.sqrt(self.pl.B2))
        dR = [np.gradient(R) for _, R, _ in self.contours]
        dZ = [np.gradient(Z) for _, _, Z in self.contours]
        dl = [np.hypot(dr, dz) for dr, dz in zip(dR, dZ, strict=True)]
        self.arc_length_coords = [np.cumsum(dl_row) for dl_row in dl]

    @property
    def pg(self) -> PsiGrid:
        return self.m.PsiGrid

    @property
    def pl(self) -> Plasma:
        return self.m.Plasma

    def grid_to_flux(self, quantity: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        interp = RegularGridInterpolator(
            (np.asarray(self.pg.Z), np.asarray(self.pg.R)), quantity
        )
        return [interp((z, r)) for _, r, z in self.contours]

    def flux_surface_average(
        self, quantity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.flux_surface_average_flux_quantity(self.grid_to_flux(quantity))

    def flux_surface_average_flux_quantity(
        self, quantity: list[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        # TODO: Assert on the shape of the quantity
        flux_integrals: list[np.float64] = [  # type: ignore
            np.trapezoid(quantity / bp, x=arc_length)
            for quantity, bp, arc_length in zip(
                quantity, self.B_p, self.arc_length_coords, strict=True
            )
        ]
        return np.array(flux_integrals) / np.asarray(self.pl.Volp_pr)

    def rmin(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.coord_extrema(np.argmin, 0)

    def zmin(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.coord_extrema(np.argmin, 1)

    def rmax(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.coord_extrema(np.argmax, 0)

    def zmax(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.coord_extrema(np.argmax, 1)

    def coord_extrema(
        self, extrema_func: Callable[[NDArray[np.float64]], int], coord_idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert coord_idx in (0, 1), "coord_idx must be 0 for R or 1 for Z"
        arg_extrema = [
            (extrema_func([R, Z][coord_idx]), R, Z) for _, R, Z in self.contours
        ]
        return (
            np.array([R[arg_extrema] for arg_extrema, R, _ in arg_extrema]),
            np.array([Z[arg_extrema] for arg_extrema, _, Z in arg_extrema]),
        )
