import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import json
import time
import os
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# MEDIAPIPE SETUP
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# VOLUME CONTROL SETUP
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]
# CAMERA SETUP
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)
# FINGER PATTERN → ALPHABET MAPPING
alphabet_patterns = {
    "A": [0,1,0,1,0],
    "B": [0,0,0,1,0],
    "C": [0,0,0,1,1],
    "D": [1,0,1,0,1],
    "E": [0,0,1,0,1],
    "F": [0,0,1,1,0],
    "G": [0,0,1,1,1],
    "H": [0,1,0,0,0],
    "I": [0,1,0,0,1],
    "J": [0,1,0,1,0],
    "K": [0,1,0,1,1],
    "L": [0,1,1,0,0],
    "M": [0,0,0,0,0],
    "N": [0,0,0,0,1],
    "O": [0,1,1,1,1],
    "P": [0,0,1,0,0],
    "Q": [1,0,0,0,1],
    "R": [1,0,0,1,0],
    "S": [1,0,0,0,0],
    "T": [1,0,1,0,0],
    "U": [1,1,1,1,1],
    "V": [1,0,1,1,0],
    "W": [1,0,1,1,1],
    "X": [1,1,0,0,0],
    "Y": [1,1,0,0,1],
    "Z": [1,1,0,1,0]
}
# FINGER STATUS FUNCTION (unchanged)
def get_fingers(lm):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(1 if lm[4][1] > lm[3][1] else 0)

    # 4 other fingers
    for i in range(1, 5):
        fingers.append(1 if lm[tip_ids[i]][2] < lm[tip_ids[i]-2][2] else 0)

    return fingers
# PREVENT MULTIPLE TRIGGERS (existing flags kept)
pause_ready = True
play_ready = True
next_ready = True
prev_ready = True
mute_ready = True
unmute_ready = True

# additional ready/debounce flags for our added features
left_click_ready = True
right_click_ready = True
double_click_time = 0.4
last_left_click_time = 0
double_click_candidate = False
keyboard_type_ready = True
keyboard_type_cooldown = 0.8
last_typed_time = 0
zoom_ready = True
zoom_last_distance = None
zoom_cooldown = 0.5
zoom_last_time = 0
screenshot_ready = True
# MODES & CALIBRATION
mouse_mode = False
keyboard_mode = False
volume_mode = False
zoom_mode = False
# calibration params (default identity mapping)
calibration = {
    "src_w": 1280,
    "src_h": 720,
    "x_off": 0,
    "y_off": 0,
    "scale_x": 1.0,
    "scale_y": 1.0
}

CALIB_FILE = "calib.json"
# load calibration if exists
if os.path.exists(CALIB_FILE):
    try:
        with open(CALIB_FILE, "r") as f:
            calibration.update(json.load(f))
        print("Loaded calibration from", CALIB_FILE)
    except Exception as e:
        print("Failed to load calibration:", e)

# smoothing for cursor
prev_x, prev_y = 0, 0
smoothing = 7.0  # higher = more smooth (but lag)

# helper: map camera coords to screen coords using calibration
def camera_to_screen(cx, cy):
    screen_w, screen_h = pyautogui.size()
    # normalize
    nx = cx / calibration["src_w"]
    ny = cy / calibration["src_h"]
    # apply scale and offsets
    sx = int((nx * calibration["scale_x"] + calibration["x_off"]) * screen_w)
    sy = int((ny * calibration["scale_y"] + calibration["y_off"]) * screen_h)
    return sx, sy

def calc_dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# small helper to save calibration
def save_calibration():
    with open(CALIB_FILE, "w") as f:
        json.dump(calibration, f)
    print("Saved calibration:", calibration)

def calibrate(cap, hands):
    text = ["Calibration Step 1/2: SHOW OPEN hand (max distance) and press 'n'",
            "Calibration Step 2/2: SHOW PINCHED hand (thumb+index touching) and press 'n'"]
    vals = []
    step = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img)
        disp = frame.copy()
        cv2.putText(disp, text[step], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if res.multi_hand_landmarks:
            h, w, _ = disp.shape
            hand = res.multi_hand_landmarks[0]
            x1, y1 = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
            x2, y2 = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
            d = calc_dist((x1, y1), (x2, y2))
            cv2.circle(disp, (x1, y1), 8, (0, 255, 0), -1)
            cv2.circle(disp, (x2, y2), 8, (0, 255, 0), -1)
            cv2.line(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, f"d={int(d)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('n'):
            if res and res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                h, w, _ = disp.shape
                x1, y1 = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
                x2, y2 = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
                vals.append(calc_dist((x1, y1), (x2, y2)))
                step += 1
                if step >= 2:
                    break
            else:
                print("No hand detected. Try again.")
        elif k == ord('q'):
            break
    cv2.destroyWindow("Calibration")
    if len(vals) == 2:
        pinch_max, pinch_min = max(vals[0], vals[1]), min(vals[0], vals[1])
        calibration['pinch_min'] = max(5.0, pinch_min)
        calibration['pinch_max'] = max(pinch_min + 10.0, pinch_max)
        print("Calibration done:", calibration)
        save_calibration()
    else:
        print("Calibration incomplete.")
# MAIN LOOP
with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=2) as hands:

    while True:
        success, img = cam.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lmList = []
        lmList2 = []  # second hand if present

        # Get landmarks
        if results.multi_hand_landmarks:
            # draw all hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # primary hand (first)
            hand = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            for id, lm in enumerate(hand.landmark):
                lmList.append([id, int(lm.x * w), int(lm.y * h)])

            # if second hand present
            if len(results.multi_hand_landmarks) > 1:
                hand2 = results.multi_hand_landmarks[1]
                for id, lm in enumerate(hand2.landmark):
                    lmList2.append([id, int(lm.x * w), int(lm.y * h)])

        if lmList:
            fingers = get_fingers(lmList)  # unchanged function call

            # --- MODE: VOLUME/MEDIA (v) ---
            if volume_mode:
                # VOLUME CONTROL (Thumb + Index)
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                length = math.hypot(x2 - x1, y2 - y1)

                vol = np.interp(length, [20, 200], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)
                volBar = np.interp(length, [40, 220], [400, 150])
                volPer = np.interp(length, [40, 220], [0, 100])

                # draw HUD bar
                cv2.rectangle(img, (30, 100), (100, 400), (50, 50, 50), 2)
                cv2.rectangle(img, (30, int(volBar)), (100, 400), (0, 200, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)}%', (25, 450),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 0), 3)

                last_action = f"Volume: {int(volPer)}%"

                # BINARY GESTURE ACTIONS (existing)
                # PAUSE → [0,1,0,0,0]
                if fingers == [0,1,0,0,0] and pause_ready:
                    pyautogui.press("playpause")
                    print("⏯ Pause")
                    pause_ready = False
                if fingers != [0,1,0,0,0]:
                    pause_ready = True

                # PLAY → [1,0,0,0,0]
                if fingers == [1,0,0,0,0] and play_ready:
                    pyautogui.press("playpause")
                    print("⏯ Play ")
                    play_ready = False
                if fingers != [1,0,0,0,0]:
                    play_ready = True

                # PREVIOUS → [0,0,1,0,0]
                if fingers == [0,0,1,0,0] and prev_ready:
                    pyautogui.press("prevtrack")
                    print("👈 Previous Song")
                    prev_ready = False
                if fingers != [0,0,1,0,0]:
                    prev_ready = True

                # NEXT → [0,0,0,0,1]
                if fingers == [0,0,0,0,1] and next_ready:
                    pyautogui.press("nexttrack")
                    print("👉 Next Song")
                    next_ready = False
                if fingers != [0,0,0,0,1]:
                    next_ready = True

                # MUTE → [0,0,0,0,0]
                if fingers == [0,0,0,0,0] and mute_ready:
                    pyautogui.press("volumemute")
                    print("🔇 Mute")
                    mute_ready = False
                if fingers != [0,0,0,0,0]:
                    mute_ready = True

                # UNMUTE → [1,1,1,1,1]
                if fingers == [1,1,1,1,1] and unmute_ready:
                    pyautogui.press("volumemute")
                    print("🔊 unmute")
                    unmute_ready = False
                if fingers != [1,1,1,1,1]:
                    unmute_ready = True

                # when in volume_mode we skip all other modes' actions for safety
                cv2.putText(img, "MODE: VOLUME/MEDIA", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # --- MODE: MOUSE (m) ---
            if mouse_mode and not volume_mode:
                # Use index tip to control cursor
                ix, iy = lmList[8][1], lmList[8][2]
                sx, sy = camera_to_screen(ix, iy)

                # smoothing
                cur_x = prev_x + (sx - prev_x) / smoothing
                cur_y = prev_y + (sy - prev_y) / smoothing
                prev_x, prev_y = cur_x, cur_y

                pyautogui.moveTo(int(cur_x), int(cur_y))

                # detect left-click: thumb-index pinch
                tx, ty = lmList[4][1], lmList[4][2]
                pinch_dist = math.hypot(tx - ix, ty - iy)

                # left click when pinch close
                if pinch_dist < 40 and left_click_ready:
                    # single/double logic
                    now = time.time()
                    if double_click_candidate and (now - last_left_click_time) <= double_click_time:
                        pyautogui.doubleClick()
                        print("Mouse: Double Click")
                        double_click_candidate = False
                        left_click_ready = False
                    else:
                        pyautogui.click()
                        print("Mouse: Left Click")
                        double_click_candidate = True
                        last_left_click_time = now
                        left_click_ready = False
                if pinch_dist >= 45:
                    left_click_ready = True
                    # reset double-click candidate after time
                    if double_click_candidate and (time.time() - last_left_click_time) > double_click_time:
                        double_click_candidate = False

                # right click: pinch index+middle (index tip to middle tip)
                mid_x, mid_y = lmList[12][1], lmList[12][2]
                idx_mid_dist = math.hypot(ix - mid_x, iy - mid_y)
                if idx_mid_dist < 35 and right_click_ready:
                    pyautogui.click(button='right')
                    print("Mouse: Right Click")
                    right_click_ready = False
                if idx_mid_dist >= 40:
                    right_click_ready = True
                # SCROLLING — index + middle vertical movement (new)
                if idx_mid_dist < 50:
                    # compare vertical positions: index (iy) vs middle (mid_y)
                    # iy < mid_y -> index is higher on the image => user moved up -> scroll up
                    # iy > mid_y -> index is lower -> scroll down
                    if iy < (mid_y - 20):
                        pyautogui.scroll(20)  # scroll up
                        # small print to terminal for debugging
                        print("Mouse: Scroll Up")
                        # small delay to prevent insanely fast scrolling
                        time.sleep(0.08)
                    elif iy > (mid_y + 20):
                        pyautogui.scroll(-20)  # scroll down
                        print("Mouse: Scroll Down")
                        time.sleep(0.08)

                # screenshot gesture: all five fingers extended (open palm) + hold for a second
                if fingers == [1,0,0,0,1]:
                    if screenshot_ready:
                        screenshot_ready = False
                        # take screenshot
                        fname = f"screenshot_{int(time.time())}.png"
                        img_s = pyautogui.screenshot()
                        img_s.save(fname)
                        print("Screenshot saved:", fname)
                else:
                    screenshot_ready = True

                cv2.putText(img, "MODE: MOUSE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # --- MODE: KEYBOARD (k) ---
            if keyboard_mode and not volume_mode:
                # detect alphabet via your existing mapping and type the character
                detected = ""
                for char, pattern in alphabet_patterns.items():
                    if fingers == pattern:
                        detected = char
                        break

                if detected != "":
                    cv2.putText(img, f"Alphabet: {detected}", (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

                    now = time.time()
                    if keyboard_type_ready and (now - last_typed_time) > keyboard_type_cooldown:
                        pyautogui.press(detected.lower())
                        print("Typed:", detected)
                        keyboard_type_ready = False
                        last_typed_time = now
                if 'pattern' in locals():
                    if fingers != pattern:
                        keyboard_type_ready = True

                cv2.putText(img, "MODE: KEYBOARD (A-Z)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,180,0), 2)

            # --- MODE: ZOOM (p) ---
            if zoom_mode and not volume_mode:
                # require two hands for pinch-zoom; use index tip distance between handsp
                if lmList2:
                    x1, y1 = lmList[8][1], lmList[8][2]       # hand1 index
                    x2, y2 = lmList2[8][1], lmList2[8][2]    # hand2 index
                    dist = math.hypot(x2 - x1, y2 - y1)

                    if zoom_last_distance is None:
                        zoom_last_distance = dist

                    # if distance increased significantly -> zoom out or in depending on mapping
                    now = time.time()
                    if now - zoom_last_time > zoom_cooldown:
                        if dist - zoom_last_distance > 40:
                            # zoom out (Ctrl + '-')
                            pyautogui.hotkey('ctrl', '-')
                            print("Zoom Out")
                            zoom_last_time = now
                        elif zoom_last_distance - dist > 40:
                            pyautogui.hotkey('ctrl', '+')
                            print("Zoom In")
                            zoom_last_time = now
                        zoom_last_distance = dist

                cv2.putText(img, "MODE: ZOOM", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (180,0,255), 2)

            # If neither volume_mode nor other functional mode is active
            # we can still show alphabet (optional)
            if not (keyboard_mode or volume_mode or mouse_mode or zoom_mode):
                # ORIGINAL ALPHABET DETECTION (kept as display-only if no modes)
                detected = ""
                for char, pattern in alphabet_patterns.items():
                    if fingers == pattern:
                        detected = char
                        break
                if detected != "":
                    cv2.putText(img, f"Alphabet: {detected}",
                                (30, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 255, 255), 3)
        # FRONTEND LABELS (always draw current mode labels)
        y0 = 30
        dy = 30
        # base label
        cv2.putText(img, "Gesture Controller", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        if mouse_mode:
            cv2.putText(img, "MOUSE (m)", (20, y0 + dy * 0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(img, "MOUSE (m)", (20, y0 + dy * 0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

        if keyboard_mode:
            cv2.putText(img, "KEYBOARD (k)", (20, y0 + dy * 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,180,0), 2)
        else:
            cv2.putText(img, "KEYBOARD (k)", (20, y0 + dy * 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

        if volume_mode:
            cv2.putText(img, "VOLUME (v)", (20, y0 + dy * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            cv2.putText(img, "VOLUME (v)", (20, y0 + dy * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

        # calibration
        if calibration:
            cv2.putText(img, "CALIBRATION (c)", (20, y0 + dy * 3),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
        
        if zoom_mode:
            cv2.putText(img, "ZOOM (p)", (20, y0 + dy * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,0,255), 2)
        else:
            cv2.putText(img, "ZOOM (p)", (20, y0 + dy * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

        # show last action text if present
        try:
            if last_action:
                cv2.putText(img, str(last_action), (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        except NameError:
            pass
        # NORMAL WINDOW (not fullscreen)
        cv2.imshow("Gesture Music Controller", img)

        # KEY HANDLING
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            # toggles + actions
            if key == ord('q'):
                # quit
                break
            elif key == ord('m'):
                mouse_mode = not mouse_mode
                # when mouse_mode toggled on, disable conflicting modes
                if mouse_mode:
                    keyboard_mode = False
                    volume_mode = False
                    zoom_mode = False
                print("Mouse mode:", mouse_mode)
            elif key == ord('k'):
                keyboard_mode = not keyboard_mode
                if keyboard_mode:
                    mouse_mode = False
                    volume_mode = False
                    zoom_mode = False
                print("Keyboard mode (A-Z):", keyboard_mode)
            elif key == ord('v'):
                volume_mode = not volume_mode
                if volume_mode:
                    mouse_mode = False
                    keyboard_mode = False
                    zoom_mode = False
                print("Volume/Media mode:", volume_mode)
            # (REMOVED) 'e' key handling - removed per request
            elif key == ord('p'):
                zoom_mode = not zoom_mode
                if zoom_mode:
                    mouse_mode = False
                    keyboard_mode = False
                    volume_mode = False
                print("Zoom mode:", zoom_mode)
            elif key == ord('s'):
                # save calibration to file
                save_calibration()
            elif key == ord('c'): 
                calibrate(cam, hands)   
            elif key == ord('S'):
                # allow uppercase 'S' as well if needed
                save_calibration()
            # else: ignore other keys

    cam.release()
    cv2.destroyAllWindows()
