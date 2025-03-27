from imports import *
from util_func import *

if __name__ == "__main__":

    # set subject number
    subject = 999
    dir_data = "../data"
    f_name = f"sub_{subject}_data.csv"
    full_path = os.path.join(dir_data, f_name)

    # Uncomment to check if file already exists
    if os.path.exists(full_path):
        print(f"File {f_name} already exists. Aborting.")
        sys.exit()

    condition_1 = {
        'name':
        "congruent_pointer_middle",
        'context': ["S", "S", "D", "D"],
        'effector': ["L1", "R1", "L2", "R2"],
        'resp_key': [pygame.K_f, pygame.K_j, pygame.K_d, pygame.K_k],
        'stim_region': ["A", "B", "A", "B"],
        'cue_img': [
            pygame.image.load("../images/cue_pointer_middle_2.png"),
            pygame.image.load("../images/cue_pointer_middle_2.png"),
            pygame.image.load("../images/cue_pointer_middle_1.png"),
            pygame.image.load("../images/cue_pointer_middle_1.png")
        ]
    }

    condition_2 = {
        "name":
        "incongruent_pointer_middle",
        "context": ["S", "S", "D", "D"],
        "effector": ["L1", "R1", "L2", "R2"],
        'resp_key': [pygame.K_f, pygame.K_j, pygame.K_d, pygame.K_k],
        "stim_region": ["A", "B", "B", "A"],
        'cue_img': [
            pygame.image.load("../images/cue_pointer_middle_2.png"),
            pygame.image.load("../images/cue_pointer_middle_2.png"),
            pygame.image.load("../images/cue_pointer_middle_1.png"),
            pygame.image.load("../images/cue_pointer_middle_1.png")
        ]
    }

    condition_3 = {
        "name":
        "congruent_pinky_thumb",
        "context": ["S", "S", "D", "D"],
        "effector": ["L1", "R1", "L2", "R2"],
        "resp_key": [pygame.K_c, pygame.K_m, pygame.K_q, pygame.K_p],
        "stim_region": ["A", "B", "A", "B"],
        'cue_img': [
            pygame.image.load("../images/cue_pinky_thumb_1.png"),
            pygame.image.load("../images/cue_pinky_thumb_1.png"),
            pygame.image.load("../images/cue_pinky_thumb_2.png"),
            pygame.image.load("../images/cue_pinky_thumb_2.png")
        ]
    }

    condition_4 = {
        "name":
        "incongruent_pinky_thumb",
        "context": ["S", "S", "D", "D"],
        "effector": ["L1", "R1", "L2", "R2"],
        "resp_key": [pygame.K_c, pygame.K_m, pygame.K_q, pygame.K_p],
        "stim_region": ["A", "B", "B", "A"],
        'cue_img': [
            pygame.image.load("../images/cue_pinky_thumb_1.png"),
            pygame.image.load("../images/cue_pinky_thumb_1.png"),
            pygame.image.load("../images/cue_pinky_thumb_2.png"),
            pygame.image.load("../images/cue_pinky_thumb_2.png")
        ]
    }

    condition_list = [condition_1, condition_2, condition_3, condition_4]
    condition = pd.DataFrame(condition_list[(subject - 1) % len(condition_list)])

    ds = make_stim_cats()

    # plot the stimuli coloured by label
    # fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    # sns.scatterplot(data=ds, x="x", y="y", hue="cat", alpha=0.5, ax=ax[0, 0])
    # sns.scatterplot(data=ds, x="xt", y="yt", hue="cat", alpha=0.5, ax=ax[0, 1])
    # ax[0, 0].plot([0, 100], [0, 100], 'k--')
    # ax[0, 1].plot([0, 5], [0, np.pi / 2], 'k--')
    # ax[0, 0].set_title(condition['name'][0])
    # plt.show()

    # plot_stim_space_examples(ds)

    # Initialize Pygame
    pygame.init()

    # useful constants but need to change / verify on each computer
    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54

    # grating size
    size_cm = 5
    size_px = int(size_cm * px_per_cm)

    # set small window potentially useful for debugging
    # screen_width, screen_height = 800, 600
    # center_x = screen_width // 2
    # center_y = screen_height // 2
    # screen = pygame.display.set_mode((screen_width, screen_height))

    # set full screen
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h
    center_x = screen_width // 2
    center_y = screen_height // 2
    screen = pygame.display.set_mode((screen_width, screen_height),
                                     pygame.FULLSCREEN)

    # Hide the mouse cursor
    pygame.mouse.set_visible(False)

    # Set up fonts
    font = pygame.font.Font(None, 36)

    # Define colors
    black = (0, 0, 0)
    white = (255, 255, 255)
    grey = (126, 126, 126)
    green = (0, 255, 0)
    red = (255, 0, 0)

    # create clocks to keep time
    clock_state = pygame.time.Clock()
    clock_exp = pygame.time.Clock()

    time_state = 0.0
    time_exp = 0.0

    # set the current state to the initial state
    state_current = "state_init"

    # behavioural measurements
    resp = -1
    rt = -1

    # trial counter
    trial = -1
    n_trial = ds.shape[0]

    # choose 1 or 2 randomly
    sub_task = np.random.choice([1, 2])

    # record keeping
    trial_data = {
        'condition': [],
        'subject': [],
        'trial': [],
        'sub_task': [],
        'cat': [],
        'x': [],
        'y': [],
        'xt': [],
        'yt': [],
        'resp': [],
        'rt': [],
        'fb': []
    }

    running = True
    while running:

        time_exp += clock_exp.tick()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    keep_running = False
                    pygame.quit()
                else:
                    resp = event.key

        if state_current == "state_init":
            time_state += clock_state.tick()

            text = font.render("Please press the space bar to begin", True,
                               (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.fill(black)
            screen.blit(text, text_rect)

            if resp == pygame.K_SPACE:
                time_state = 0
                state_current = "state_iti"

        if state_current == "state_finished":
            time_state += clock_state.tick()
            text = font.render("You finished! Thank you for being awesome!", True,
                               (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.fill(black)
            screen.blit(text, text_rect)

        if state_current == "state_iti":
            time_state += clock_state.tick()
            screen.fill(black)
            pygame.draw.line(screen, white, (center_x, center_y - 10),
                             (center_x, center_y + 10), 4)
            pygame.draw.line(screen, white, (center_x - 10, center_y),
                             (center_x + 10, center_y), 4)
            if time_state > 1000:
                resp = -1
                rt = -1
                time_state = 0
                trial += 1
                if trial == n_trial - 1:
                    state_current = "state_finished"
                else:
                    sf = ds['xt'].iloc[trial] * (px_per_cm**-1)
                    ori = ds['yt'].iloc[trial]
                    cat = ds['cat'].iloc[trial]
                    sub_task = np.random.choice([1, 2])
                    state_current = "state_cue"

        if state_current == "state_cue":
            time_state += clock_state.tick()
            screen.fill(black)

            if sub_task == 1:
                cue_img = condition.loc[(condition["context"] == "S"), 'cue_img'].values[0]
            elif sub_task == 2:
                cue_img = condition.loc[(condition["context"] == "D"), 'cue_img'].values[0]

            screen.blit(cue_img, (center_x - cue_img.get_width() / 2, center_y - cue_img.get_height() / 2))

            if time_state > 2000:
                time_state = 0
                resp = -1
                state_current = "state_stim"

        if state_current == "state_stim":
            time_state += clock_state.tick()
            screen.fill(black)

            if sub_task == 1:
                pygame.draw.rect(screen, grey,
                                 (center_x - 2 * size_px / 2, center_y -
                                  2 * size_px / 2, 2 * size_px, 2 * size_px))
            else:
                pygame.draw.polygon(screen, grey,
                                    [(center_x, center_y - 1.3 * size_px),
                                     (center_x + 1.3 * size_px, center_y),
                                     (center_x, center_y + 1.3 * size_px),
                                     (center_x - 1.3 * size_px, center_y)])

            grating_patch = create_grating_patch(size_px, sf, ori)
            grating_surface = grating_to_surface(grating_patch)
            screen.blit(grating_surface,
                        (center_x - size_px / 2, center_y - size_px / 2))

            if sub_task == 1:
                resp_key_context = condition.loc[condition["context"] == "S", "resp_key"]

            elif sub_task == 2:
                resp_key_context = condition.loc[condition["context"] == "D", "resp_key"]

            if np.isin(resp, resp_key_context):
                rt = time_state
                time_state = 0

                if sub_task == 1:

                    resp_corr = condition.loc[
                        (condition["context"] == "S")
                        & (condition["stim_region"] == ["A", "B"][cat - 1][0]),
                        'resp_key'].values[0]

                elif sub_task == 2:

                    resp_corr = condition.loc[
                        (condition["context"] == "D")
                        & (condition["stim_region"] == ["A", "B"][cat - 1][0]),
                        'resp_key'].values[0]

                if resp == resp_corr:
                    fb = "Correct"
                else:
                    fb = "Incorrect"

                state_current = "state_feedback"

        if state_current == "state_feedback":
            time_state += clock_state.tick()

            if fb == "Correct":
                pygame.draw.circle(screen, green, (center_x, center_y),
                                   size_px / 2 + 10, 5)

            elif fb == "Incorrect":
                pygame.draw.circle(screen, red, (center_x, center_y),
                                   size_px / 2 + 10, 5)

            if time_state > 1000:
                trial_data['condition'].append(condition['name'][0])
                trial_data['subject'].append(subject)
                trial_data['trial'].append(trial)
                trial_data['sub_task'].append(sub_task)
                trial_data['x'].append(np.round(ds.x[trial], 2))
                trial_data['y'].append(np.round(ds.y[trial], 2))
                trial_data['xt'].append(np.round(sf, 2))
                trial_data['yt'].append(np.round(ori, 2))
                trial_data['cat'].append(resp_corr)
                trial_data['resp'].append(resp)
                trial_data['rt'].append(rt)
                trial_data['fb'].append(fb)
                pd.DataFrame(trial_data).to_csv(full_path, index=False)
                time_state = 0
                state_current = "state_iti"

        pygame.display.flip()
