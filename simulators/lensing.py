import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
from astropy.convolution import convolve, Gaussian2DKernel

from simulators.units import *
from simulators.lensing_profiles import MassProfileSIE, MassProfileNFW, LightProfileSersic

import jax
import jax.numpy as jnp

from collections import defaultdict
from functools import partial


class LensingSim:
    def __init__(self, lenses_list=[{}], subs_list=[{}], sources_list=[{}], global_dict={}, observation_dict={}):
        """
        Class for simulation of strong lensing images
        """

        self.lenses_list = lenses_list
        self.subs_list = subs_list
        self.sources_list = sources_list

        self.global_dict = global_dict
        self.observation_dict = observation_dict

        self.set_up_global()
        self.set_up_observation()

    def set_up_global(self):
        """Set some global variables so don't need to recompute each time"""
        self.z_s = self.global_dict["z_s"]
        self.z_l = self.global_dict["z_l"]

        self.D_s = Planck15.angular_diameter_distance(z=self.z_s).value  # * Mpc
        self.D_l = Planck15.angular_diameter_distance(z=self.z_l).value  # * Mpc

        self.Sigma_crit = 1.0 / (4 * jnp.pi * GN) * self.D_s / ((self.D_s - self.D_l) * self.D_l) / M_s

    def set_up_observation(self):
        """Set up observational grid and parameters"""
        # Coordinate limits (in arcsecs)
        self.theta_x_lims = self.observation_dict["theta_x_lims"]
        self.theta_y_lims = self.observation_dict["theta_y_lims"]

        # Size of grid
        self.n_x = self.observation_dict["n_x"]
        self.n_y = self.observation_dict["n_y"]

        # Exposure and background noise level
        self.exposure = self.observation_dict["exposure"]
        self.f_iso = self.observation_dict["f_iso"]

        # x/y-coordinates of grid and pixel area in arcsec**2

        self.theta_x, self.theta_y = jnp.meshgrid(jnp.linspace(self.theta_x_lims[0], self.theta_x_lims[1], self.n_x), jnp.linspace(self.theta_y_lims[0], self.theta_y_lims[1], self.n_y))

        self.x, self.y = self.D_l * self.theta_x * asctorad, self.D_l * self.theta_y * asctorad

        self.pix_area = ((self.theta_x_lims[1] - self.theta_x_lims[0]) / self.n_x) * ((self.theta_y_lims[1] - self.theta_y_lims[0]) / self.n_y)

    @partial(jax.jit, static_argnums=(0,))
    def lensed_image(self):
        """Get strongly lensed image"""

        # Get lensing potential gradients

        x_d, y_d = jnp.zeros((self.n_x, self.n_y)), jnp.zeros((self.n_x, self.n_y))

        for lens_dict in self.lenses_list:
            _x_d, _y_d = MassProfileSIE(
                x_0=lens_dict["theta_x_0"] * self.D_l * asctorad,
                y_0=lens_dict["theta_y_0"] * self.D_l * asctorad,
                r_E=lens_dict["theta_E"] * self.D_l * asctorad,
                q=lens_dict["q"],
            ).deflection(self.x, self.y)
            x_d += _x_d
            y_d += _y_d

#         subs_dict = defaultdict(list)

#         for d in self.subs_list:
#             for key, value in d.items():
#                 subs_dict[key].append(value)

        # for key, value in d.items():
        #     if key != "profile":
        #         subs_dict[key] = jnp.array(subs_dict[key])

        # _x_d, _y_d = MassProfileNFW(
        #     x_0=subs_dict["theta_x_0"] * self.D_l * asctorad,
        #     y_0=subs_dict["theta_y_0"] * self.D_l * asctorad,
        #     M_200=subs_dict["M_200"],
        #     kappa_s=(subs_dict["rho_s"]) * (subs_dict["r_s"]) / self.Sigma_crit / Mpc,
        #     r_s=subs_dict["r_s"],
        # ).deflection(self.x, self.y)
        # x_d += jnp.sum(_x_d, axis=-1)
        # y_d += jnp.sum(_y_d, axis=-1)

        # TODO: vmap over
        for lens_dict in self.subs_list:
            _x_d, _y_d = MassProfileNFW(
                x_0=lens_dict["theta_x_0"] * self.D_l * asctorad,
                y_0=lens_dict["theta_y_0"] * self.D_l * asctorad,
                M_200=lens_dict["M_200"],
                kappa_s=(lens_dict["rho_s"]) * (lens_dict["r_s"]) / self.Sigma_crit / Mpc,
                r_s=lens_dict["r_s"],
            ).deflection(self.x, self.y)

            x_d += _x_d
            y_d += _y_d

        # Evaluate source image on deflected lens plane to get lensed image

        f_lens = jnp.zeros((self.n_x, self.n_y))

        for source_dict in self.sources_list:
            if source_dict["profile"] == "Sersic":

                f_lens += (
                    LightProfileSersic(
                        x_0=source_dict["theta_x_0"] * self.D_l * asctorad,
                        y_0=source_dict["theta_y_0"] * self.D_l * asctorad,
                        S_tot=source_dict["S_tot"],
                        r_e=source_dict["theta_e"] * self.D_l * asctorad,
                        n_srsc=source_dict["n_srsc"],
                    ).flux(self.x - x_d, self.y - y_d)
                    * self.D_l**2
                    / radtoasc**2
                )
            else:
                raise Exception("Unknown source profile specification!")

        f_iso = self.f_iso * jnp.ones((self.n_x, self.n_y))  # Isotropic background
        i_tot = (f_lens + f_iso) * self.exposure * self.pix_area  # Total lensed image

        return i_tot
