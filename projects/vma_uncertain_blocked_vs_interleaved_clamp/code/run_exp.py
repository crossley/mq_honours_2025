import sys
import os
import serial
import time
import struct
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    subject = 1
    dir_data = "../data"
    f_name = f"sub_{subject}_data.csv"
    full_path = os.path.join(dir_data, f"sub_{subject}_data.csv")
    full_path_move = os.path.join(dir_data, f"sub_{subject}_data_move.csv")

    # Uncomment to check if file already exists
    if os.path.exists(full_path):
        print(f"File {f_name} already exists. Aborting.")
        sys.exit()

    use_liberty = False


    # This method grabs the position of the sensor
    def getPosition(ser, recordsize, averager):
        ser.reset_input_buffer()

        # Set variables
        # This defines the length of the binary header (bytes 0-7)
        header = 8
        # This defines the bytesize of IEEE floating point
        byte_size = 4

        # Obtain data
        ser.write(b"P")
        # time.sleep(0.1)
        # print("inWaiting " + str(ser.inWaiting()))
        # print("recorded size " + str(recordsize))

        # Read header to remove it from the input buffer
        ser.read(header)

        positions = []

        # Read the three coordinates
        for x in range(3):
            # Read the coordinate
            coord = ser.read(byte_size)

            # Convert hex to floating point (little endian order)
            coord = struct.unpack("<f", coord)[0]

            positions.append(coord)

        return positions


    if use_liberty:
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = "COM3"

        print(ser)
        ser.open()

        # Checks serial port if open
        if ser.is_open == False:
            print("Error! Serial port is not open")
            exit()

        # Send command to receive data through port
        ser.write(b"P")
        time.sleep(1)

        # Checks if Liberty is responding(e.g on)
        if ser.inWaiting() < 1:
            print("Error! Check if liberty is on!")
            exit()

        # Set liberty output mode to binary
        ser.write(b"F1\r")
        time.sleep(1)

        # Set distance unit to centimeters
        ser.write(b"U1\r")
        time.sleep(0.1)

        # Set hemisphere to +Z
        ser.write(b"H1,0,0,1\r")
        time.sleep(0.1)

        # Set sample rate to 240hz
        ser.write(b"R4\r")
        time.sleep(0.1)

        # Reset frame count
        ser.write(b"Q1\r")
        time.sleep(0.1)

        # Set output to only include position (no orientation)
        ser.write(b"O1,3,9\r")
        time.sleep(0.1)
        ser.reset_input_buffer()

        # Obtain data
        ser.write(b"P")
        time.sleep(0.1)

        # Size of response
        recordsize = ser.inWaiting()
        ser.reset_input_buffer()
        averager = 4

    # useful constants but need to change / verify on each computer
    # lab computer is resolution 1920 x 1080
    # monitor size is 60 cm x 33 cm
    # px_per_cm = np.mean([1920 / 60, 1080 / 33])
    px_per_cm = 1080 / 33

    n_trial = 430

    condition_list = ["blocked", "interleaved"]
    condition = condition_list[(subject - 1) % 2]

    low_su = 0.000000000001 * px_per_cm
    high_su = 0.025 * px_per_cm

    su_interleaved = np.random.choice([low_su * px_per_cm, high_su * px_per_cm],
                                      n_trial)

    su_1 = low_su * px_per_cm * np.ones(n_trial // 2)
    su_2 = high_su * px_per_cm * np.ones(n_trial // 2)
    if np.random.rand() > 0.5:
        su_blocked = np.concatenate((su_1, su_2))
    else:
        su_blocked = np.concatenate((su_2, su_1))

    if condition == "interleaved":
        su = su_interleaved
    elif condition == "blocked":
        su = su_blocked

    su[:30] = su_interleaved[:30]

    rotation = np.zeros(n_trial)
    rotation[30:130] = 15 * np.pi / 180
    rotation[230:330] = 15 * np.pi / 180

    endpoint_visible = np.ones(n_trial)
    endpoint_visible[130:180] = 0
    endpoint_visible[330:380] = 0

#    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(10, 5))
#    ax[0, 0].plot(su / su.max(), label='sensory uncertainty')
#    ax[1, 0].plot(rotation / rotation.max(), label='rotation')
#    ax[2, 0].plot(endpoint_visible, label='endpoint visible')
#    [x.legend() for x in ax.flatten()]
#    plt.show()

    pygame.init()

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
    grey = (128, 128, 128)
    white = (255, 255, 255)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    yellow = (255, 255, 0)
    orange = (255, 165, 0)
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    # cursor circle radius
    cursor_radius = 8
    start_radius = 15
    target_radius = 15

    n_points = 20

    # relevant coords
    center_x = screen.get_width() // 2
    center_y = screen.get_height() // 2

    start_pos = (center_x, center_y + 2 * px_per_cm)
    target_pos = (center_x, center_y - 6 * px_per_cm)

    # create clocks to keep time
    clock_state = pygame.time.Clock()
    clock_exp = pygame.time.Clock()

    t_state = 0.0
    time_exp = 0.0

    # initial state
    state_init = "state_init"

    # set the current state to the initial state
    state_current = state_init

    # behavioural measurements
    rt = -1
    mt = -1
    ep = -1
    resp = -1

    # record keeping
    trial_data = {
        'condition': [],
        'subject': [],
        'trial': [],
        'su': [],
        'rotation': [],
        'rt': [],
        'mt': [],
        'ep': []
    }

    trial_move = {
        'condition': [],
        'subject': [],
        'trial': [],
        'state': [],
        't': [],
        'x': [],
        'y': []
    }

    if use_liberty == False:

        # set the current state to the initial state
        state_current = "state_init"

    else:

        rig_coord_upper_left = (0, 0)
        rig_coord_upper_right = (0, 0)
        rig_coord_lower_right = (0, 0)
        rig_coord_lower_left = (0, 0)

        min_x = 1
        max_x = 2
        min_y = 1
        max_y = 2

        calibrating = True
        state_current = "calibrate_upper_left"
        while calibrating:

            screen.fill((0, 0, 0))

            hand_pos = getPosition(ser, recordsize, averager)[0:2]

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        pygame.quit()
                    else:
                        resp = event.key

            if state_current == "calibrate_upper_left":
                t_state += clock_state.tick()
                text = font.render(
                    "Please move to the upper left corner of the screen", True,
                    (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 2))
                screen.fill(black)
                screen.blit(text, text_rect)

                pos_x = 0 + screen_width / 4
                pos_y = 0 + screen_height / 4
                pos = (pos_x, pos_y)
                pygame.draw.circle(screen, white, pos, 15, 0)

                if resp == pygame.K_SPACE:
                    resp = []
                    rig_coord_upper_left = hand_pos
                    state_current = "calibrate_upper_right"

            if state_current == "calibrate_upper_right":
                t_state += clock_state.tick()
                text = font.render(
                    "Please move to the upper right corner of the screen", True,
                    (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 2))
                screen.fill(black)
                screen.blit(text, text_rect)

                pos_x = screen.get_width() - screen_width / 4
                pos_y = screen_height / 4
                pos = (pos_x, pos_y)
                pygame.draw.circle(screen, white, pos, 15, 0)

                if resp == pygame.K_SPACE:
                    resp = []
                    rig_coord_upper_right = hand_pos
                    state_current = "calibrate_lower_right"

            if state_current == "calibrate_lower_right":
                t_state += clock_state.tick()
                text = font.render(
                    "Please move to the lower right corner of the screen", True,
                    (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 2))
                screen.fill(black)
                screen.blit(text, text_rect)

                pos_x = screen_width - screen_width / 4
                pos_y = screen_height - screen_height / 4
                pos = (pos_x, pos_y)
                pygame.draw.circle(screen, white, pos, 15, 0)

                if resp == pygame.K_SPACE:
                    resp = []
                    rig_coord_lower_right = hand_pos
                    state_current = "calibrate_lower_left"

            if state_current == "calibrate_lower_left":
                t_state += clock_state.tick()
                text = font.render(
                    "Please move to the lower left corner of the screen", True,
                    (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 2))

                screen.fill(black)
                screen.blit(text, text_rect)

                pos_x = screen_width / 4
                pos_y = screen_height - screen_height / 4
                pos = (pos_x, pos_y)
                pygame.draw.circle(screen, white, pos, 15, 0)

                if resp == pygame.K_SPACE:
                    resp = []
                    rig_coord_lower_left = hand_pos
                    state_current = "state_init"
                    calibrating = False

                    x_ul, y_ul = rig_coord_upper_left
                    x_ur, y_ur = rig_coord_upper_right
                    x_ll, y_ll = rig_coord_lower_left
                    x_lr, y_lr = rig_coord_lower_right

                    min_x = min(x_ul, x_ur, x_ll, x_lr)
                    max_x = max(x_ul, x_ur, x_ll, x_lr)
                    min_y = min(y_ul, y_ur, y_ll, y_lr)
                    max_y = max(y_ul, y_ur, y_ll, y_lr)

            flipped_screen = pygame.transform.flip(screen, False, True)
            screen.blit(flipped_screen, (0, 0))
            pygame.display.update()

    # set trials / phases
    trial = 1

    running = True
    while running:

        time_exp += clock_exp.tick()
        screen.fill((0, 0, 0))

        rot_mat = np.array([[np.cos(rotation[trial]), -np.sin(rotation[trial])],
                            [np.sin(rotation[trial]),
                             np.cos(rotation[trial])]])

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit()
                else:
                    resp = event.key

        if use_liberty:
            hand_pos = getPosition(ser, recordsize, averager)[0:2]

            x = hand_pos[0]
            y = hand_pos[1]

            x = -((x - min_x) / (max_x - min_x)) + 1
            y = ((y - min_y) / (max_y - min_y)) + 0

            x = x * screen_width
            y = y * screen_height

            hand_pos = (x, y)

        else:
            hand_pos = pygame.mouse.get_pos()

        cursor_pos = np.dot(np.array(hand_pos) - np.array(start_pos),
                            rot_mat) + start_pos

        if state_current == "state_init":
            t_state += clock_state.tick()
            text = font.render("Please press the space bar to begin", True,
                               (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.fill(black)
            screen.blit(text, text_rect)

            if resp == pygame.K_SPACE:
                t_state = 0
                resp = -1
                state_current = "state_searching_ring"

        if state_current == "state_finished":
            t_state += clock_state.tick()
            text = font.render("You finished! Thank you for being awesome!", True,
                               (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.fill(black)
            screen.blit(text, text_rect)

        if state_current == "state_iti":
            t_state += clock_state.tick()
            screen.fill(black)
            if t_state > 1000:
                resp = -1
                rt = -1
                t_state = 0
                trial += 1
                if trial == n_trial:
                    state_current = "state_finished"
                else:
                    state_current = "state_searching_ring"

        if state_current == "state_searching_ring":
            t_state += clock_state.tick()

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, white, start_pos, r, 2)

            if r < 2 * start_radius:
                t_state = 0
                state_current = "state_searching_cursor"

        if state_current == "state_searching_cursor":
            t_state += clock_state.tick()

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, white, hand_pos, cursor_radius)

            if r < start_radius:
                t_state = 0
                state_current = "state_holding"
            elif r >= 2 * start_radius:
                t_state = 0
                state_current = "state_searching_ring"

        if state_current == "state_holding":
            t_state += clock_state.tick()

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            # smoothly transition from blue to red with
            # increasing time until next state
            if t_state < 2000:
                proportion = t_state / 2000
                red_component = int(255 * proportion)
                blue_component = int(255 * (1 - proportion))
                state_color = (red_component, 0, blue_component)
                pygame.draw.circle(screen, state_color, start_pos, start_radius)
                pygame.draw.circle(screen, white, hand_pos, cursor_radius)

            if r >= start_radius:
                t_state = 0
                state_current = "state_searching_cursor"

            elif t_state > 2000:
                rt = t_state
                t_state = 0
                t_state_2 = 0
                state_current = "state_ready_to_move"

        if state_current == "state_ready_to_move":
            t_state += clock_state.tick()

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, red, target_pos, target_radius)

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            if r >= start_radius:
                rt = t_state
                t_state = 0
                t_state_2 = 0
                state_current = "state_moving"

        if state_current == "state_moving":
            t_state += clock_state.tick()

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, red, target_pos, target_radius)

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            r_target = np.sqrt((target_pos[0] - start_pos[0])**2 +
                               (target_pos[1] - start_pos[1])**2)

            if r >= r_target:
                ep = cursor_pos

                ep_theta = np.arctan2(ep[1] - start_pos[1], ep[0] - start_pos[0])
                ep_target = (r_target * np.cos(ep_theta) + start_pos[0],
                             r_target * np.sin(ep_theta) + start_pos[1])

                # NOTE: Consult with Dave about what design he wants
                # ep_theta_clamp = rotation[trial]
                ep_theta_clamp = -90.0 * np.pi / 180.0
                ep_target_clamp = (r_target * np.cos(ep_theta_clamp) +
                                   start_pos[0],
                                   r_target * np.sin(ep_theta_clamp) +
                                   start_pos[1])

                mt = t_state

                cloud = np.random.multivariate_normal(
                    ep_target_clamp, [[su[trial]**2, 0], [0, su[trial]**2]],
                    n_points)

                t_state = 0
                state_current = "state_feedback_ep"

        if state_current == "state_feedback_ep":
            t_state += clock_state.tick()

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, red, target_pos, target_radius)

            if endpoint_visible[trial]:
                for i in range(n_points):
                    pygame.draw.circle(screen, white, cloud[i], cursor_radius)

            if t_state > 1000:
                trial_data['condition'].append(condition)
                trial_data['subject'].append(subject)
                trial_data['trial'].append(trial)
                trial_data['su'].append(np.round(su[trial], 2))
                trial_data['rotation'].append(np.round(rotation[trial], 2))
                trial_data['rt'].append(rt)
                trial_data['mt'].append(mt)
                trial_data['ep'].append(np.round(ep_theta, 2))
                pd.DataFrame(trial_data).to_csv(full_path, index=False)
                pd.DataFrame(trial_move).to_csv(full_path_move, index=False)
                t_state = 0
                state_current = "state_iti"

        trial_move['condition'].append(condition)
        trial_move['subject'].append(subject)
        trial_move['trial'].append(trial)
        trial_move['state'].append(state_current)
        trial_move['t'].append(time_exp)
        trial_move['x'].append(hand_pos[0])
        trial_move['y'].append(hand_pos[1])

        if use_liberty:
            flipped_screen = pygame.transform.flip(screen, False, True)
            screen.blit(flipped_screen, (0, 0))
            pygame.display.update()
        else:
            pygame.display.flip()

    pygame.quit()
