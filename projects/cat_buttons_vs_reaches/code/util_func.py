from imports import *


def make_stim_cats():

    n_stimuli_per_category = 200

    # Define covariance matrix parameters
    var = 100
    corr = 0.9
    sigma = np.sqrt(var)

    # Rotation matrix
    theta = 45 * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Standard deviations along major and minor axes
    std_major = sigma * np.sqrt(1 + corr)
    std_minor = sigma * np.sqrt(1 - corr)

    def sample_within_ellipse(mean, n_samples):

        # Sample radius
        r = np.sqrt(np.random.uniform(
            0, 9, n_samples))  # 3 standard deviations, squared is 9

        # Sample angle
        angle = np.random.uniform(0, 2 * np.pi, n_samples)

        # Convert polar to Cartesian coordinates
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Scale by standard deviations
        x_scaled = x * std_major
        y_scaled = y * std_minor

        # Apply rotation
        points = np.dot(rotation_matrix, np.vstack([x_scaled, y_scaled]))

        # Translate to mean
        points[0, :] += mean[0]
        points[1, :] += mean[1]

        return points.T

    # Generate stimuli
    category_A_mean = [40, 60]
    category_B_mean = [60, 40]

    stimuli_A = sample_within_ellipse(category_A_mean, n_stimuli_per_category)
    stimuli_B = sample_within_ellipse(category_B_mean, n_stimuli_per_category)

    # Define the labels
    labels_A = np.array([1] * n_stimuli_per_category)
    labels_B = np.array([2] * n_stimuli_per_category)

    # Concatenate the stimuli and labels
    stimuli = np.concatenate([stimuli_A, stimuli_B])
    labels = np.concatenate([labels_A, labels_B])

    # Put the stimuli and labels together into a dataframe
    ds = pd.DataFrame({'x': stimuli[:, 0], 'y': stimuli[:, 1], 'cat': labels})

    # Add a transformed version of the stimuli
    # let xt map x from [0, 100] to [0, 5]
    # let yt map y from [0, 100] to [0, 90]
    ds['xt'] = ds['x'] * 5 / 100
    ds['yt'] = (ds['y'] * 90 / 100) * np.pi / 180

    # shuffle rows of ds
    ds = ds.sample(frac=1).reset_index(drop=True)

    return ds


def create_grating_patch(size, freq, theta):
    """
    Generate a grating patch with a circular mask using NumPy.
    The units of size are pixels, the units of freq are
    cycles per pixel, and the units of theta are radians.
    """
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    x, y = np.meshgrid(x, y)

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # grating formula
    psi = 0
    gb = np.cos(2 * np.pi * freq * x_theta + psi)

    # Circular mask
    radius = size / 2
    circle_mask = (x**2 + y**2) <= radius**2
    gb *= circle_mask

    return gb


def grating_to_surface(grating_patch):
    normalized_patch = (grating_patch + 1) / 2 * 255
    uint8_patch = normalized_patch.astype(np.uint8)
    surface = pygame.Surface((grating_patch.shape[0], grating_patch.shape[1]),
                             pygame.SRCALPHA)
    pygame.surfarray.blit_array(surface, np.dstack([uint8_patch] * 3))
    return surface


def plot_stim_space_examples(ds):

    screen_width, screen_height = 800, 600
    center_x = screen_width // 2
    center_y = screen_height // 2

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Simple Category Learning')

    screen.fill((126, 126, 126))

    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54
    size_cm = 3
    size_px = int(size_cm * px_per_cm)

    # grating size
    size_cm = 3
    size_px = int(size_cm * px_per_cm)

    # range of spatial frequency
    freqs = [ds['xt'].min(), ds['xt'].mean(), ds['xt'].max()]
    freqs = [x * px_per_cm**-1 for x in freqs]

    # range of orientation
    thetas = [ds['yt'].min(), ds['yt'].mean(), ds['yt'].max()]

    # Generate and blit grating patches
    for i, freq in enumerate(freqs):
        for j, theta in enumerate(thetas):
            grating_patch = create_grating_patch(size_px, freq, theta)
            grating_surface = grating_to_surface(grating_patch)
            x = i * (screen_width / 3) + (screen_width / 6) - (size_px / 2)
            y = j * (screen_height / 3) + (screen_height / 6) - (size_px / 2)
            screen.blit(grating_surface, (x, y))

    pygame.display.flip()

    # Wait until window is closed
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit()
                if event.key == pygame.K_SPACE:
                    running = False
