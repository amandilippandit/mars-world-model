"""Mars physical constants used across the simulation, scene, and renderer.

Source notes:
- Gravity, mass, radius: NASA Mars Fact Sheet (nssdc.gsfc.nasa.gov)
- Atmosphere: Mars Climate Database; Viking & MSL surface measurements
- Solar: TOA irradiance scaled by mean orbital distance (1.524 AU)
- Sky color: approximation from MSL/Perseverance Mastcam-Z white-balanced imagery
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarsConstants:
    # ── Gravity & body ──────────────────────────────────────────────
    gravity_m_s2: float = 3.721
    radius_km: float = 3389.5
    mass_kg: float = 6.4171e23
    mu_m3_s2: float = 4.282837e13  # standard gravitational parameter (G·M)

    # ── Atmosphere (surface, mean) ──────────────────────────────────
    surface_pressure_pa: float = 610.0          # ~0.6% of Earth
    surface_density_kg_m3: float = 0.020        # ~1/60th of Earth
    surface_temperature_k: float = 210.0        # mean; ranges 130–290 K
    atm_composition: tuple = (
        ("CO2", 0.9532),
        ("N2",  0.0270),
        ("Ar",  0.0160),
        ("O2",  0.0013),
        ("CO",  0.0008),
    )
    speed_of_sound_m_s: float = 240.0           # cold thin CO2

    # ── Solar / lighting ────────────────────────────────────────────
    solar_irradiance_w_m2: float = 590.0        # ~43% of Earth's 1361
    sun_angular_diameter_arcmin: float = 21.0   # vs Earth's ~32
    mean_orbital_distance_au: float = 1.524

    # Approx. sky luminance color (linear sRGB) when looking up at zenith
    # under typical low-tau dust loading. Sunset reverses to blue near sun.
    sky_color_zenith_rgb: tuple = (0.83, 0.50, 0.30)   # butterscotch
    sky_color_horizon_rgb: tuple = (0.95, 0.65, 0.40)  # peachy
    sun_disk_color_rgb: tuple = (0.95, 0.85, 0.75)     # paler than Earth's
    typical_optical_depth_tau: float = 0.5             # range 0.1 (clear) – 5 (dust storm)

    # ── Surface / regolith ──────────────────────────────────────────
    regolith_albedo: float = 0.17               # geometric mean
    regolith_color_rgb: tuple = (0.55, 0.34, 0.22)
    static_friction_regolith: float = 0.65
    kinetic_friction_regolith: float = 0.55
    regolith_density_kg_m3: float = 1500.0      # bulk

    # ── Time ────────────────────────────────────────────────────────
    sol_seconds: float = 88775.244              # 24h 39m 35.244s
    year_sols: float = 668.6
    year_earth_days: float = 686.97

    # ── Common landmark (Perseverance landing site) ─────────────────
    jezero_lat_deg: float = 18.4447
    jezero_lon_deg: float = 77.4508             # east-positive (areocentric)
    jezero_elev_m: float = -2540.0              # relative to areoid


MARS = MarsConstants()
