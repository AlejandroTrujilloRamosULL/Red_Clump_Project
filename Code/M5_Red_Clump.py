# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:27:50 2025

@author: aleja
"""

"Example Code of the Selection Criteria and Distance Modulus Calculation of one"
"of the clusters under study, in this case M5"

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from scipy.stats import gaussian_kde

"Opening and extracting of data in the file retrieved from Gaia"

# Opening 
with fits.open("path_to_your_file") as hdul:
    hdul.info()
    bin_header = hdul[1].header
    data = hdul[1].data
    
# Close plot
plt.close("all")
    
# Convert to native-endian arrays, if not seaborn does not work
def to_native(arr):
    arr = np.array(arr)
    if arr.dtype.byteorder not in ('=', '|'):
        arr = arr.byteswap().newbyteorder()
    return arr

# Applying native data to all data retrieved so KDE plotting works, 
# retrieving the parameter values from the data array
color       = to_native(data["bp_rp"]) # [mag]
g_mag       = to_native(data["phot_g_mean_mag"]) # [mag]
pm          = to_native(data["pm"]) # [mas/yr]
metallicity = to_native(data["mh_gspphot"])
teff        = to_native(data["teff_gspphot"])
logg        = to_native(data["logg_gspphot"])
parallax    = to_native(data["parallax"]) # [mas]
excess      = to_native(data["phot_bp_rp_excess_factor"])
astro_noise = to_native(data["astrometric_excess_noise"])
parallax_error = to_native(data["parallax_error"]) # [mas]
parallax_over_error = to_native(data["parallax_over_error"]) # [mas]
variable_flag = to_native(data["phot_variable_flag"])
ruwe = to_native(data["ruwe"])

# Masking of parameters to get rid of nan values
mask_color = np.isnan(color)
mask_mag = np.isnan(g_mag)
mask_pm = np.isnan(pm)
mask_z = np.isnan(metallicity)
mask_excess = np.isnan(excess)
mask_teff = np.isnan(teff)
mask_logg = np.isnan(logg)
mask_astro_noise = np.isnan(astro_noise)
mask_parallax = np.isnan(parallax)
mask_parallax_error = np.isnan(parallax_error)
mask_parallax_over_error = np.isnan(parallax_over_error)
mask_ruwe = np.isnan(ruwe)
mask_variable = (variable_flag == "VARIABLE")

# Averages and standard deviations of some parameters
median_parallax = np.nanmedian(parallax)
std_parallax = np.nanstd(parallax)
median_pm = np.nanmedian(pm)
avg_pm = np.nanmean(pm)
avg_z = np.nanmean(metallicity)
std_pm = np.nanstd(pm)
std_z = np.nanstd(metallicity)

"Masking and filtering to select the stars of the cluster"

# Masking applied to filter for stars in the cluster
gaia_data_ult = np.vstack((color, g_mag, ruwe, parallax, parallax_over_error))
mask_nan = mask_color | mask_mag 
mask_astrometric_excess_noise = (astro_noise < 1)
mask_ruwe_range = (ruwe < 1.4)
mask_parallax_range = (parallax > median_parallax - 1*std_parallax) & (parallax < median_parallax + 1*std_parallax)
mask_phot_bp_rp = (excess < 1.3)
mask_pm_range_2 = (pm > median_pm - 1*std_pm) & (pm < median_pm + 1*std_pm)
mask_parallax_error_range = (parallax_over_error > 4)
total_mask = mask_ruwe_range & mask_parallax_range & mask_phot_bp_rp & mask_parallax_error_range & mask_pm_range_2 & ~mask_nan
gaia_data_ultimate = gaia_data_ult[:, total_mask==True]

# Masking and selecting color, magnitude and parallax for cluster stars and 
color_filt  = color[total_mask]
g_mag_filt  = g_mag[total_mask]
parallax_filt = parallax[total_mask]

# Distance and absolute magnitudes for filtered sample of cluster stars
distance_pc = 1000/parallax_filt # [pc]
abs_mag_filt = g_mag_filt - 5 * np.log10(distance_pc / 10)

# Masking for RR Lyrae stars
variable_flag_filt = variable_flag[total_mask]
mask_rrlyrae = np.char.strip(variable_flag_filt) == "VARIABLE"
color_rrlyrae = color_filt[mask_rrlyrae]
abs_mag_rrlyrae = abs_mag_filt[mask_rrlyrae]

# Masking for Kiel Diagram
gaia_data_teff_logg = np.vstack((teff, logg))
total_mask_teff_logg = mask_teff | mask_logg
gaia_data_teff_logg = gaia_data_teff_logg[:, total_mask_teff_logg==False]

# Calculate distance in parsecs from parallax for selected cluster stars
distance_pc = 1000/gaia_data_ultimate[3] # [pc]

# Stacking the color, magnitude and parallax for all the stars 
# (non-filtered case)
gaia_data_no_filter = np.vstack((color, g_mag, parallax))
mask_no_filter = mask_color | mask_mag | mask_parallax
gaia_data_no_filter = gaia_data_no_filter[:, mask_no_filter==False]

# Calculation of absolute magnitude of non-filtered stars
abs_mag = gaia_data_ultimate[1] - 5*(np.log10(distance_pc/10))
abs_mag_no_filter = gaia_data_no_filter[1] - 5*(np.log10(((1000/gaia_data_no_filter[2]))/10))

"CMD region masking for differentiating the different type of stars in the Horizontal Branch"

# We define a Horizontal Branch limiting range and define the red clump mask 
# inside of the branch
hb_mag_min = -0.5
hb_mag_max = 2.8
hb_mask = (abs_mag_filt > hb_mag_min) & (abs_mag_filt < hb_mag_max)
red_clump_hb_mask = (abs_mag_filt > hb_mag_min) & (abs_mag_filt < 2)

# Selecting the median color in the RR Lyrae group of stars
rr_median_color = np.nanmedian(color_rrlyrae)

# Maskng for BHB and RC group of stars
blue_hb_mask = hb_mask & (color_filt < rr_median_color)
red_clump_mask = red_clump_hb_mask & (color_filt > rr_median_color) & (color_filt < 0.98)

"Calculation of the distance modulus of the Cluster"

# Filtering parameters for red clump selection 
rc_g = g_mag_filt[red_clump_mask]
rc_abs_mag = abs_mag_filt[red_clump_mask]
rc_parallax = parallax_filt[red_clump_mask]

# Rejecting nan values for color and magnitude
valid_rc = (~np.isnan(rc_g)) & (~np.isnan(rc_abs_mag))
rc_g = rc_g[valid_rc]
rc_M = rc_abs_mag[valid_rc]
rc_parallax = rc_parallax[valid_rc]

# Defining the dstance modulus for its calculation
distance_modulus = rc_g - rc_abs_mag
rc_distance_pc = 1000/rc_parallax
distance_modulus_from_parallax = 5.0*np.log10(rc_distance_pc/10.0)

# Retrieving mean and standard deviations for the distance modulus from 
# absolute magnitude and apparent magnitude subtraction
mean_dm = np.nanmean(distance_modulus) 
std_dm = np.nanstd(distance_modulus)  

# Retrieving mean and standard deviations for the distance modulus from parallax
mean_dm_parallax = np.nanmean(distance_modulus_from_parallax) 
std_dm_parallax = np.nanstd(distance_modulus_from_parallax) 

# Print results
print(f"Number of Red Clump stars used: {len(distance_modulus)}")
print(f"Mean distance modulus (m - M): {mean_dm:.3f} mag  (std = {std_dm:.3f}")
print(f"Mean distance modulus from parallax distances: {mean_dm_parallax:.3f} mag (std = {std_dm_parallax:.3f})")
print()

"Plotting snippet of code"

plt.figure(1)
plt.scatter(gaia_data_ultimate[0], abs_mag, color="lightcoral", zorder=2, edgecolors="black", s=50, alpha=0.9, label="Rest of Cluster Stars")
plt.scatter(color_rrlyrae, abs_mag_rrlyrae, color="gold", zorder=4, edgecolors="black", s=50, label="Variable Stars")
plt.scatter(color_filt[blue_hb_mask], abs_mag_filt[blue_hb_mask], zorder=3, color='blue', s=50, edgecolors='black', label="Blue HB")
plt.scatter(color_filt[red_clump_mask], abs_mag_filt[red_clump_mask], zorder=3, color='red', s=50, edgecolors='black', label="Red Clump")
plt.scatter(gaia_data_no_filter[0], abs_mag_no_filter, color="lightgray", zorder=1, edgecolors="black", alpha=0.4, label="Field Stars")
plt.xlabel(r"$G_{BP}$ - $G_{RP}$" + " [mag]", fontsize=20)
plt.ylabel(r"$M_{G}$" + " [mag]", fontsize=20)
plt.gca().invert_yaxis()
plt.tick_params(direction="in", which="major", length=8, labelsize=18, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=6, labelsize=18, top=True, right=True)
plt.xticks(np.arange(-4, 4, step=0.5), minor=True)
plt.yticks(np.arange(24, 10, step=1), minor=True)
plt.xlim(-0.5, 3.5)
plt.ylim(16, -8)
plt.tight_layout()
plt.legend(loc="best", prop={"size":13}, fontsize=18)
plt.show()


# Plotting
plt.figure(2)
hb = plt.hexbin(gaia_data_ultimate[0], abs_mag, gridsize=220, bins="log", mincnt=1)
plt.xlabel(r"$G_{BP} - G_{RP}$" + " [mag]", fontsize=26, labelpad=10)
plt.ylabel(r"$M_{G}$" + " [mag]", fontsize=26)
plt.gca().invert_yaxis()
plt.tick_params(direction="in", which="major", length=6, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", labelsize=24, top=True, right=True)
plt.xticks(np.arange(-4, 4, step=0.5), minor=True)
plt.yticks(np.arange(24, 10, step=1), minor=True)
plt.xlim(-1, 3)
#plt.ylim(8, -4)
plt.title("M5 Globular Cluster", fontsize=28)
cbar = plt.colorbar(hb, pad=0.)
cbar.ax.tick_params(labelsize="large")
plt.tight_layout()
plt.show()

"""
# Plotting Kiel Diagram
plt.figure(3)
plt.scatter(gaia_data_teff_logg[0], gaia_data_teff_logg[1], color="blue", edgecolors="black", s=30)
plt.tick_params(direction="in", which="major", top=True, right=True)
plt.tick_params(direction="in", which="minor", top=True, right=True)
plt.xlim(3000, 8000)
plt.ylim(2, 5)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.show()
"""

# Plotting for KDE in Horizontal Branch

# Masking for absolute magnitude in the Horizontal Branch
mask_hb_mag = (abs_mag > -0.5) & (abs_mag < 2.0)
gaia_data_ultimate_hb = gaia_data_ultimate[:, mask_hb_mag==True]

# Horizontal branch color
hb_color = gaia_data_ultimate_hb[0]
sorted_color = np.sort(color_filt)

# Defining bw so KDE becomes smooth
def kde_with_bw_adjust(data, bw_adjust=1.0):
    data = np.asarray(data)
    kde = gaussian_kde(data)
    scott_bw = kde.factor
    kde.set_bandwidth(bw_method=scott_bw*bw_adjust)
    return kde

bw_adj = 0.3

# Defining the KDE regions
kde = kde_with_bw_adjust(hb_color, bw_adjust=bw_adj)
x_blue = np.arange(np.sort(hb_color)[0] - 1, color_rrlyrae.min(), step=0.001)
x = np.arange(color_rrlyrae.min(), np.sort(color_rrlyrae)[14], step=0.001)
x_redclump = np.arange(np.sort(color_rrlyrae)[14], 0.95, step=0.001)
x_rest = np.arange(0.95,  np.sort(hb_color)[-1] + 1, step=0.001)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

# Plot on the top (CMD of M5 with defined Horizontal Branch)
ax1.scatter(gaia_data_ultimate[0], abs_mag,
            color="lightcoral", zorder=1, edgecolors="black", s=50, alpha=0.8)
ax1.scatter(color_rrlyrae, abs_mag_rrlyrae,
            color="gold", zorder=2, edgecolors="black", s=50, label="Variable Stars")
ax1.scatter(color_filt[blue_hb_mask], abs_mag_filt[blue_hb_mask],
            color='blue', s=50, edgecolors='black', label="Blue HB")
ax1.scatter(color_filt[red_clump_mask], abs_mag_filt[red_clump_mask],
            color='red', s=50, edgecolors='black', label="Red Clump")

ax1.set_ylabel(r"$M_{G}$ [mag]", fontsize=20)
ax1.invert_yaxis()
ax1.tick_params(direction="in", which="major", length=8, labelsize=20, top=True, right=True)
ax1.tick_params(direction="in", which="minor", length=6, labelsize=20, top=True, right=True)
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(12, -4)
ax1.hlines(hb_mag_min - 0.2, xmin=sorted_color[0] - 1, xmax=sorted_color[-1] + 1, color="black", linestyle="dashed", label="Limits of Horizontal Branch")
ax1.hlines(hb_mag_min + 0.2, xmin=sorted_color[0] - 1, xmax=sorted_color[-1] + 1, color="black", linestyle="dashed")
ax1.set_title("M5 Color–Magnitude Diagram and Density Plot of Horizontal Branch", fontsize=18, pad=10)

# Plot on the bottom (KDE)
g = sns.kdeplot(hb_color, bw_adjust=bw_adj, fill=True,
                color="lightblue", linewidths=2, ax=ax2)
ax2.fill_between(x_blue, kde(x_blue), color="blue", alpha=0.6, label="Blue HB Region")
ax2.fill_between(x, kde(x), color="yellow", alpha=0.6, label="RR Lyrae Region")
ax2.fill_between(x_redclump, kde(x_redclump), color="red", alpha=0.4, label="Red Clump Region")
ax2.fill_between(x_rest, kde(x_rest), color="lightcoral", alpha=0.3, label="Rest")
ax2.set_xlabel(r"$G_{BP} - G_{RP}$ [mag]", fontsize=20)
ax2.set_ylabel("Probability Density", fontsize=20)
ax2.tick_params(direction="in", which="major", top=True, right=True, labelsize=20)
ax2.tick_params(direction="in", which="minor", top=True, right=True, labelsize=20)
ax2.set_xlim(-0.5, 2.5)
ax2.legend(loc="best", fontsize=16)
plt.tight_layout()
plt.show()

"Histogram study of parallax and proper motion for the cluster selection"

# Histogram study of parallax and proper motion
parallax_bins = np.arange(min(parallax[mask_parallax==False]), max(parallax[mask_parallax==False]), 0.05) 
pm_bins = np.arange(min(pm[mask_pm==False]), max(pm[mask_pm==False]), 0.2) 

parallax_filt = parallax[total_mask]
pm_filt = pm[total_mask]

plt.figure(5)
counts_parallax, bins_parallax, patches = plt.hist(parallax[mask_parallax==False], bins=parallax_bins)
sns.histplot(parallax[mask_parallax==False], bins=parallax_bins, kde=False, color="slateblue", edgecolor="black", linewidth=2)
for val, color_line, label in [(median_parallax, "red", fr"Median $\varpi$ = {median_parallax:.3f} [mas]"),
                               (median_parallax + std_parallax, "darkorange", r"Median $\varpi$ + $\sigma_{\varpi}$ = " + f"{median_parallax + std_parallax:.3f} [mas]"),
                               (median_parallax - std_parallax, "darkorange", r"Median $\varpi$ - $\sigma_{\varpi}$ = " + f"{median_parallax - std_parallax:.3f} [mas]")]:
    bin_index = np.searchsorted(bins_parallax, val) - 1
    bin_index = np.clip(bin_index, 0, len(counts_parallax)-1)
    plt.vlines(val, 0, counts_parallax[bin_index], colors=color_line, linestyles="dashed", label=label, linewidth=2)
plt.tick_params(direction="in", which="major", length=6, top=True, right=True, labelsize=24)
plt.xlabel(r"Parallax $\varpi$ [mas]", fontsize=30)
plt.ylabel("Counts", fontsize=30)
plt.legend(loc="best", prop={"size":16})
plt.xlim(-3, 3)
plt.ylim(0.1)
plt.show()

plt.figure(6)
counts_pm, bins_pm, patches = plt.hist(pm[mask_pm==False], bins=pm_bins)
sns.histplot(pm[mask_pm==False], bins=pm_bins, kde=False, color="slateblue", edgecolor="black", linewidth=2)
for val, color_line, label in [(median_pm, "red", fr"Median $\mu$ = {median_pm:.3f} [mas/yr]"),
                               (median_pm + std_pm, "darkorange", r"Median $\mu$ + $\sigma_{\mu}$ = " + f"{median_pm + std_pm:.3f} [mas/yr]"),
                               (median_pm - std_pm, "darkorange", r"Median $\mu$ - $\sigma_{\mu}$ = " +  f"{median_pm - std_pm:.3f} [mas/yr]")]:
    bin_index = np.searchsorted(bins_pm, val) - 1
    bin_index = np.clip(bin_index, 0, len(counts_pm)-1)
    plt.vlines(val, 0, counts_pm[bin_index], colors=color_line, linestyles="dashed", label=label, linewidth=2)
plt.tick_params(direction="in", which="major", length=6, top=True, right=True, labelsize=24)
plt.xlabel(r"Proper Motion $\mu$ [mas/yr]", fontsize=30)
plt.ylabel("Counts", fontsize=30)
plt.xlim(0, 20)
plt.ylim(0.1)
plt.legend(loc="best", prop={"size":16})
plt.show()

