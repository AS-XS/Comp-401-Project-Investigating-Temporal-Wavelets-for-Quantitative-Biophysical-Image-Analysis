#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# In[2]:


#Set random seed for reproducibility
np.random.seed(42)


# In[3]:


# Simulation parameters
grid_size_x = 100  # size of the x
grid_size_y = 100   # size of the y
space_size = 100 # physical space size
num_emitters = 1000  # number of point emitters (proteins)
timesteps = 50  # number of time steps for diffusion simulation
resolution = 100  # resolution of the continuous space for better smoothing
prefactor = np.sqrt(2)  # prefactor for Brownian motion in 2D
diffusion_coefficient = 1.0  # diffusion coefficient (D) in µm^2/s
time_step = 0.02  # time step (dt) in seconds
psf_sigma = 1.0  # standard deviation for the Gaussian PSF
noise_sigma = 0.05  # standard deviation for Gaussian noise
brightness_factor = 1000 # Adjust this factor as needed to simulate realistic photon counts
molecule_magnitude = 10  # Define the magnitude of each molecule

# Define the mesh confinement boundary
confine_x_min, confine_x_max = -(grid_size_x + (0.1*grid_size_x)), (grid_size_x + (0.1*grid_size_x))  # x-axis confinement boundaries
confine_y_min, confine_y_max = -(grid_size_y + (0.1*grid_size_y)), (grid_size_y + (0.1*grid_size_y))  # y-axis confinement boundaries

# Mesh parameters within general confinement region
mesh_width = confine_x_max - confine_x_min
mesh_height = confine_y_max - confine_y_min

#graphing area
graph_x_min = 0
graph_x_max = space_size  
graph_y_min = 0
graph_y_max = space_size

# Define the skeletal meshwork (e.g., 4x4 grid of confinement zones)
num_boxes = 10  # Number of confinement boxes along each axis
box_size_x = mesh_width / num_boxes
box_size_y = mesh_height / num_boxes
hop_probability = 0.1  # Probability of hopping to a neighboring box

# create a 2D grid to represent the point emitters
grid = np.zeros((grid_size_x, grid_size_y))


# In[4]:


# Initialize random positions for the point emitters inside the grid
emitters = np.random.uniform(0, space_size, (num_emitters, 2))

# Initialize trajectories for each molecule
trajectories = [[] for _ in range(num_emitters)]


# In[5]:


def diffuse_with_mesh_and_global_confinement(emitters, space_size, prefactor, diffusion_coefficient, time_step, num_boxes, box_size_x, box_size_y, hop_probability, confine_x_min, confine_x_max, confine_y_min, confine_y_max):
    new_emitters = []
    for emitter in emitters:
        current_x, current_y = emitter

        # Diffuse molecule (Brownian motion step)
        step_scale = prefactor * np.sqrt(diffusion_coefficient * time_step)
        dx, dy = np.random.normal(0, step_scale, 2)
        new_x = current_x + dx
        new_y = current_y + dy

        # Wrap around at global boundaries (toroidal space)
        if new_x < confine_x_min:
            new_x = confine_x_max - (confine_x_min - new_x)
        elif new_x > confine_x_max:
            new_x = confine_x_min + (new_x - confine_x_max)

        if new_y < confine_y_min:
            new_y = confine_y_max - (confine_y_min - new_y)
        elif new_y > confine_y_max:
            new_y = confine_y_min + (new_y - confine_y_max)

        # Determine the current box bounds
        box_start_x = (current_x // box_size_x) * box_size_x
        box_start_y = (current_y // box_size_y) * box_size_y
        box_end_x = box_start_x + box_size_x
        box_end_y = box_start_y + box_size_y

        # Check if the molecule tries to escape the current box
        if np.random.uniform(0, 1) < hop_probability:
            # Allow hopping into a neighboring box within global boundaries
            if new_x < box_start_x:
                new_x = max(new_x, box_start_x - box_size_x)
            elif new_x > box_end_x:
                new_x = min(new_x, box_end_x + box_size_x)

            if new_y < box_start_y:
                new_y = max(new_y, box_start_y - box_size_y)
            elif new_y > box_end_y:
                new_y = min(new_y, box_end_y + box_size_y)
        else:
            # Constrain the molecule within its current box
            new_x = np.clip(new_x, box_start_x, box_end_x)
            new_y = np.clip(new_y, box_start_y, box_end_y)

        # Append updated position
        new_emitters.append([new_x, new_y])

    return np.array(new_emitters)


# In[6]:


def rickerWavelet(t,a,b):
    t = (t-b)/a
    return (1/np.sqrt(a))*(1-t**2)*np.exp((-1/2)*(t**2))


# In[7]:


def waveletTransform(ts, signal, a):
    center = ts[len(ts) // 2]  
    kernel = rickerWavelet(ts, a, center)
    kernel = np.fft.fftshift(kernel)

    kernel_length = signal.shape[-1]
    if len(kernel) < kernel_length:
        padding = (kernel_length - len(kernel)) // 2
        kernel = np.pad(kernel, (padding, kernel_length - len(kernel) - padding), mode='constant')
    elif len(kernel) > kernel_length:
        excess = (len(kernel) - kernel_length) // 2
        kernel = kernel[excess:excess + kernel_length]

    convolution = np.fft.irfft(np.fft.rfft(signal) * np.conj(np.fft.rfft(kernel, n=signal.shape[-1])))
    return convolution


# In[8]:


def inverseWaveletTransform(ts, waveletCoefficients, scale, Cg=np.pi):
    center = ts[-1]/2
    signal = np.zeros(len(waveletCoefficients))  

    kernel = rickerWavelet(ts, scale, center)  
    kernel = np.fft.fftshift(kernel)
    signal += (1 / scale**2) * np.fft.irfft(np.fft.rfft(kernel) * np.fft.rfft(waveletCoefficients))

    return (1 / Cg) * signal


# In[9]:


# Function to compute the Mean Squared Error between the original and reconstructed signal
def compute_error(original_signal, reconstructed_signal):
    return np.mean((original_signal - reconstructed_signal) ** 2)


# In[10]:


# Function to find the best wavelet scale
def find_best_wavelet_scale(ts, signal, scales):
    best_scale = None
    min_error = float('inf')
    best_reconstructed_signal = None

    # Try different scales
    for scale in scales:
        # Apply wavelet transform
        transformed_signal = waveletTransform(ts, signal, scale)
        
        # Inverse wavelet transform to reconstruct the signal
        reconstructed_signal = inverseWaveletTransform(ts, transformed_signal, scale)
        
        # Compute the error between the original and reconstructed signals
        error = compute_error(signal, reconstructed_signal)
        
        # Update best scale if a lower error is found
        if error < min_error:
            min_error = error
            best_scale = scale
            best_reconstructed_signal = reconstructed_signal

    return best_scale, best_reconstructed_signal


# In[11]:


# Function to compute the autocorrelation of a signal
def autoCorrelation(data):
    # Nearest size with power of 2
    size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype(int)

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - np.mean(data)

    # Compute the FFT
    fft = np.fft.fft(ndata, size)

    # Get the power spectrum
    pwr = np.abs(fft) ** 2

    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr = np.fft.ifft(pwr).real / var / len(data)

    # Truncate to the original size
    acorr = acorr[:len(data)]

    return acorr


# In[12]:


# Define the hyperbolic decay model
def hyperbolic_decay(tau, a, td):
    return a / (tau / td + 1)


# In[13]:


def fit_autocorrelation_to_decay_exclude_zero(acorr, xs, beam_radius):
    # Exclude tau = 0 (first data point)
    acorr_no_zero = acorr[1:]
    xs_no_zero = xs[1:]
    
    # Fit the hyperbolic decay model
    popt, pcov = curve_fit(hyperbolic_decay, xs_no_zero, acorr_no_zero, p0=[1.0, 1.0])
    a_fit, td_fit = popt
    
    # Calculate diffusion coefficient D
    D = beam_radius**2 / (4 * td_fit)
    
    # Generate the fitted autocorrelation curve
    fitted_acorr = hyperbolic_decay(xs, a_fit, td_fit)
    
    return a_fit, td_fit, D, fitted_acorr


# In[14]:


def compute_lags(signal_length):
    return np.arange(1, signal_length + 1)


# In[15]:


def autoCorrelation_exclude_zero(data):
    size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype(int)
    var = np.var(data)
    ndata = data - np.mean(data)
    fft = np.fft.fft(ndata, size)
    pwr = np.abs(fft) ** 2
    acorr = np.fft.ifft(pwr).real / var / len(data)
    return acorr[1:]  # Exclude τ=0


# In[16]:


def fit_autocorrelation(acorr, lags, beam_radius):
    truncated_lags = lags[:len(acorr)]  
    truncated_acorr = acorr

    initial_a = np.max(truncated_acorr)  
    initial_td = np.mean(truncated_lags) / 2  
    p0 = [initial_a, initial_td]

    popt, _ = curve_fit(hyperbolic_decay, truncated_lags, truncated_acorr, p0=p0, maxfev=2000)
    a_fit, td_fit = popt

    D = beam_radius**2 / (4 * td_fit)

    return a_fit, td_fit, D


# In[17]:


def reconstruct(acorr, xs, scale, beam_radius):    
    # Fit the autocorrelation data
    acorr_no_zero = acorr[1:]  # Exclude τ=0
    xs_no_zero = xs[1:]
    popt, _ = curve_fit(hyperbolic_decay, xs_no_zero, acorr_no_zero, p0=[1.0, 1.0])
    a_fit, td_fit = popt

    # Calculate diffusion coefficient
    D = beam_radius**2 / (4 * td_fit)

    # Manually adjust the reconstruction
    reconstructed_acorr = hyperbolic_decay(xs, a_fit * scale, td_fit)

    return a_fit, td_fit, D, reconstructed_acorr


# In[18]:


# Time evolution simulation with mesh confinement
returnedArray = []
for t in range(timesteps):
    # diffuse emitters with confinement
    emitters = diffuse_with_mesh_and_global_confinement(emitters, space_size, prefactor, diffusion_coefficient, time_step, num_boxes, box_size_x, box_size_y, hop_probability,confine_x_min, confine_x_max, confine_y_min, confine_y_max)

    # Count how many molecules are inside the graphed area
    inside_graph = np.sum(
        (emitters[:, 0] >= graph_x_min) & (emitters[:, 0] <= graph_x_max) &
        (emitters[:, 1] >= graph_y_min) & (emitters[:, 1] <= graph_y_max)
    )

    # Print or store the count for each timestep
    print(f"Timestep {t+1}: {inside_graph} molecules inside the graphed area.")
    
    # create a high-resolution blank space for emitters
    high_res_space = np.zeros((resolution, resolution))
    high_res_space *= brightness_factor
    
    for i, emitter in enumerate(emitters):
        # Map continuous emitter positions to high-resolution grid
        x = int(emitter[0] * (resolution / space_size))
        y = int(emitter[1] * (resolution / space_size))

        # Clip the indices to ensure they are within bounds
        x = np.clip(x, 0, resolution - 1)
        y = np.clip(y, 0, resolution - 1)

        high_res_space[x, y] += molecule_magnitude
        trajectories[i].append(emitter)  # Append current position to trajectory

    # Convolve with Gaussian PSF, add noise, and visualize
    simulated_image = gaussian_filter(high_res_space, sigma=psf_sigma)
    noise = np.random.normal(0, noise_sigma, simulated_image.shape)
    noised_image = simulated_image + noise
    noised_image = np.abs(noised_image)
    returnedArray.append(noised_image)

    # create a figure with scatter plot and PSF-blurred scatter for comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # plot scatter plot of emitters in continuous space (left)
    axes[0].scatter(emitters[:, 1], emitters[:, 0], c='blue', s=100, alpha=0.8, label="Emitters")
    axes[0].set_xlim(0, space_size)
    axes[0].set_ylim(0, space_size)
    axes[0].invert_yaxis()  # Invert y-axis for correct orientation
    axes[0].set_title(f"Protein Positions with Confinement - Timestep {t + 1}")
    axes[0].legend()
    axes[0].grid(True)

    # plot the PSF-blurred image with added Gaussian noise (mid)
    im = axes[1].imshow(noised_image, cmap='PiYG', interpolation='bilinear', extent=(0, space_size, 0, space_size))
    axes[1].set_title(f"Simulated Image with Gaussian Noise - Timestep {t + 1}")
    plt.colorbar(im, ax=axes[1], label='Intensity')

    # Plot Mesh simulation (right)
    colors = plt.cm.viridis(np.linspace(0, 1, num_emitters))
    # Plot the skeletal meshwork
    for i in range(1, num_boxes):
        # Vertical lines
        axes[2].axvline(confine_x_min + i * box_size_x, color='yellow', linestyle='--', linewidth=1)
        # Horizontal lines
        axes[2].axhline(confine_y_min + i * box_size_y, color='yellow', linestyle='--', linewidth=1)

    # Plot trajectories
    for i, trajectory in enumerate(trajectories):
        traj_x = [pos[0] for pos in trajectory]
        traj_y = [pos[1] for pos in trajectory]
        axes[2].plot(traj_x, traj_y, color=colors[i], linewidth=1, label=f'Molecule {i+1}' if t == timesteps - 1 else "")

    # Plot emitter positions within the meshwork
    axes[2].scatter(emitters[:, 1], emitters[:, 0], c=[colors[i]], s=50)
    axes[2].set_xlim(0, space_size)
    axes[2].set_ylim(0, space_size)
    axes[2].invert_yaxis()
    axes[2].set_title(f"Skeletal Meshwork Simulation- Timestep {t + 1}")
    axes[2].grid(False)

    # Add global confinement boundaries to the mesh plot
    axes[2].axvline(confine_x_min, color='red', linestyle='--', linewidth=1, label='Global Boundary')
    axes[2].axvline(confine_x_max, color='red', linestyle='--', linewidth=1)
    axes[2].axhline(confine_y_min, color='red', linestyle='--', linewidth=1)
    axes[2].axhline(confine_y_max, color='red', linestyle='--', linewidth=1)

    # Set the background color of the mesh graph
    axes[2].set_facecolor('darkblue')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


# In[19]:


plt.imshow(returnedArray[0])


# In[20]:


transposedArray = np.transpose(returnedArray)


# In[21]:


np.shape(transposedArray)


# In[22]:


plt.plot(transposedArray[50,50])


# In[23]:


# Define a range of possible scales
scale_min = 1  # Minimum scale value
scale_max = 10  # Maximum scale value
scale_step = 0.1  # Step size for scale range

# Generate scale values to explore
scales = np.arange(scale_min, scale_max, scale_step)
xs = np.linspace(0, timesteps * time_step, timesteps)
signal = transposedArray[50, :]  


# In[24]:


transform = []

# Apply wavelet transform to each row
for i in range(signal.shape[0]):  
    row_signal = signal[i, :]
    row_transform = []
    for scale in scales:
        transformed_row_signal = waveletTransform(xs, row_signal, scale)
        row_transform.append(transformed_row_signal)
    
    transform.append(np.array(row_transform))

# Convert the list to a numpy array 
transform = np.array(transform)

# Initialize the array for the inverse transform
inverseTransformSignal = np.zeros_like(signal)

for i in range(signal.shape[0]):  # Loop over each row of the signal
    row_transform = transform[i, :, :]  # Get the transformed coefficients for this row

    # Inverse transform for each scale
    for scale_idx in range(len(scales)):
        # Ensure you are passing a 1D wavelet coefficient for each scale
        inverse_signal = inverseWaveletTransform(xs, row_transform[scale_idx], scales[scale_idx])
        
        # Store the result in the inverseTransformSignal for the current row
        inverseTransformSignal[i, :] = inverse_signal


# In[97]:


# Initialize arrays to store best scales and reconstructed signals
best_scales = []
reconstructed_signals = []

# Loop through each row in the signal
for i in range(signal.shape[0]):  
    row_signal = signal[i, :]  
    
    # Find the best scale and corresponding reconstructed signal for this row
    best_scale, best_reconstructed_signal = find_best_wavelet_scale(xs, row_signal, scales)
    
    # Store the results
    best_scales.append(best_scale)
    reconstructed_signals.append(best_reconstructed_signal)

# Convert reconstructed signals to a 2D array
reconstructed_signals = np.array(reconstructed_signals)

# Display results
print("Best scales for each row:", best_scales)

# Plot the original signal
plt.subplot(2, 1, 1)
plt.imshow(signal, aspect='auto', cmap='viridis')
plt.title("Original Signal")
plt.xlabel("Time Steps")
plt.ylabel("Rows")
plt.colorbar(label="Intensity")

# Plot the reconstructed signal
plt.subplot(2, 1, 2)
plt.imshow(reconstructed_signals, aspect='auto', cmap='viridis')
plt.title("Reconstructed Signal")
plt.xlabel("Time Steps")
plt.ylabel("Rows")
plt.colorbar(label="Intensity")

# Adjust layout and display
plt.tight_layout()
plt.show()


# In[99]:


# Select row 50 of the signal (index 49 since indexing starts from 0)
row_50_signal = signal[49, :]  # Extract row 50

# Find the best wavelet scale for this row
best_scale, best_reconstructed_signal = find_best_wavelet_scale(xs, row_50_signal, [1])

# Compute the autocorrelation of the original and reconstructed signals
original_acorr = autoCorrelation(row_50_signal)
reconstructed_acorr = autoCorrelation(best_reconstructed_signal)

# Print the best scale
print("Best scale for row 50:", best_scale)

# Plot the original and reconstructed signals
plt.figure(figsize=(12, 6))

# Original vs reconstructed signal
plt.subplot(2, 1, 1)
plt.plot(xs, row_50_signal, label='Original Signal (Row 50)', alpha=0.7)
plt.plot(xs, best_reconstructed_signal, label=f'Reconstructed Signal (Best Scale: {best_scale})', linestyle='--')
plt.title("Wavelet Transform and Reconstruction for Row 50")
plt.xlabel("X")
plt.ylabel("Signal Amplitude")
plt.legend()

# Autocorrelation comparison
plt.subplot(2, 1, 2)
plt.plot(original_acorr, label='Original Signal Autocorrelation', alpha=0.7)
plt.plot(reconstructed_acorr, label='Reconstructed Signal Autocorrelation', linestyle='--')
plt.title("Autocorrelation of Original and Reconstructed Signal (Row 50)")
plt.xlabel("Lag(timestep)")
plt.ylabel("Autocorrelation")
plt.legend()

plt.tight_layout()
plt.show()


# In[101]:


plt.plot(best_reconstructed_signal)
plt.title("Best reconstructed signal")
plt.xlabel("Time steps")
plt.ylabel("signal amlitude")
plt.show()


# In[28]:


# Parameters for simulations
beam_radius = 0.5 
row_index = 50
simulated_D_values = []
# Number of simulations
num_simulations = 10 
diffusion_coefficient = 2.5
row_signal = simulated_image[row_index, :] 
acorr = autoCorrelation_exclude_zero(row_signal) 
lags = np.arange(1, len(acorr) + 1)  
row_index = 50


# In[29]:


import scipy 
curve_fit = scipy.optimize.curve_fit


# In[30]:


# Call the function with the correct number of arguments
a_fit, td_fit, D, fitted_acorr = fit_autocorrelation_to_decay_exclude_zero(original_acorr, xs, beam_radius)

# Print the best-fit parameters
print(f"Best-fit a: {a_fit}")
print(f"Best-fit td: {td_fit}")
print(f"Calculated D: {D}")

# Plot the autocorrelation and the fitted curve
plt.figure(figsize=(8, 6))
plt.plot(xs, original_acorr, label='Original Signal Autocorrelation', alpha=0.7)
plt.plot(xs, fitted_acorr, label=f'Fitted Hyperbolic Decay (a={a_fit:.2f}, td={td_fit:.2f})', linestyle='--')
plt.title("Autocorrelation Fitting to Hyperbolic Decay")
plt.xlabel("Lag (τ)")
plt.ylabel("Autocorrelation")
plt.legend()
plt.show()


# In[31]:


lags


# In[32]:


#Single decay model
def model(t,td,a,b):
    return a/(t/td + 1) + b


# In[33]:


a_fit, td_fit, D = fit_autocorrelation(acorr, lags, beam_radius)


# In[34]:


lags


# In[35]:


#Your lags correspond to time between frames,

ts = lags[:len(reconstructed_acorr)]*0.02

#easier to tell what is going on if you just fit directly
fit,cov = curve_fit(model,ts,reconstructed_acorr, p0 = [10,1,0])


# In[103]:


plt.plot(ts,model(ts,*fit), label = 'Best Model Fit')
plt.plot(ts,reconstructed_acorr, label = 'Wavelet Autocorrelation')
plt.title("Wavelet Autocorrelation and Model Fit")
plt.xlabel("Lag (τ) in seconds")
plt.ylabel("Autocorrelation")
plt.legend()


# In[37]:


fit


# In[38]:


(beam_radius**2)/(4*fit[0])

#Your diffusion coefficient


# In[39]:


td_fit


# In[40]:


D


# In[95]:


scale = 5
# Call the function with the manual scale
a_fit, td_fit, D, reconstructed_acorr = reconstruct(original_acorr, xs, scale, beam_radius)

# Print the parameters
print(f"Best-fit a: {a_fit}")
print(f"Best-fit td: {td_fit}")
print(f"Calculated D: {D}")

# Plot the original autocorrelation and the manually scaled reconstruction
plt.figure(figsize=(8, 6))
plt.plot(xs, original_acorr, label='Original Signal Autocorrelation', alpha=0.7)
plt.plot(xs, reconstructed_acorr, label=f'Reconstructed with Scale {scale} (a={a_fit:.2f}, td={td_fit:.2f})', linestyle='--')
plt.title("Wavelet Scale Reconstruction")
plt.xlabel("Lag (τ)")
plt.ylabel("Autocorrelation")
plt.legend()
plt.show()


# In[ ]:




