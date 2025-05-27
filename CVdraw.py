import cv2
import numpy as np

# Global variables for mouse callback
ix, iy, k = 200, 200, -1

# Mouse callback function
def mouse(event, x, y, flags, param):
    global ix, iy, k
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        k = 1

# Setup window and mouse callback
cv2.namedWindow("draw")
cv2.setMouseCallback("draw", mouse)

# Start webcam capture
cap = cv2.VideoCapture(0)

# Wait for user to click and start tracking
while True:
    ret, frm = cap.read()
    if not ret:
        print("Failed to access webcam.")
        break

    frm = cv2.flip(frm, 1)
    cv2.imshow("draw", frm)

    if cv2.waitKey(1) == 27 or k == 1:  # ESC or mouse click
        old_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(frm)
        break

cv2.destroyWindow("draw")

# Initial point from mouse click
old_pts = np.array([[ix, iy]], dtype=np.float32).reshape(-1, 1, 2)
color = (0, 255, 0)
c = 0

# Start tracking loop
while True:
    ret, new_frm = cap.read()
    if not ret:
        break

    new_frm = cv2.flip(new_frm, 1)
    new_gray = cv2.cvtColor(new_frm, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        new_gray,
        old_pts,
        None,
        maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.08)
    )

    key = cv2.waitKey(1) & 0xFF

    # Handle key events
    if key == ord('e'):  # Erase drawing
        mask = np.zeros_like(new_frm)
    elif key == ord('c'):  # Change color
        color = (0, 0, 0)
        lst = list(color)
        c += 1
        lst[c % 3] = 255
        color = tuple(lst)
    elif key == ord('g'):
        pass  # Reserved for future features
    else:
        for i, j in zip(old_pts, new_pts):
            x, y = j.ravel()
            a, b = i.ravel()
            cv2.line(mask, (int(a), int(b)), (int(x), int(y)), color, 15)

    # Draw tracking point
    x, y = new_pts[0][0]
    cv2.circle(new_frm, (int(x), int(y)), 3, (255, 255, 0), 2)

    # Combine mask and frame
    blended = cv2.addWeighted(new_frm, 0.8, mask, 0.2, 0.1)

    # Display windows
    cv2.imshow("tracking", blended)
    cv2.imshow("drawing", mask)

    # Update previous frame and point
    old_gray = new_gray.copy()
    old_pts = new_pts.reshape(-1, 1, 2)

    if key == 27:  # ESC key to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
