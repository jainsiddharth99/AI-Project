from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "thanos",
    4: "snake",
    5: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]



def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "thanos":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"
        if move2 == "rock":
            return "User"
        if move2 == "snake":
            return "Computer"


    if move1 == "snake":
        if move2 == "scissors":
            return "Computer"
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"
        if move2 == "thanos":
            return "User"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"
        if move2 == "thanos":
            return "Computer"
        if move2 == "snake":
            return "User"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"
        if move2 == "thanos":
            return "User"
        if move2 == "snake":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"
        if move2 == "thanos":
            return "Computer"
        if move2 == "snake":
            return "User"





model = load_model("aigame.h5")

cap = cv2.VideoCapture(0)
cap.set(3, 1280) # 3 - PROPERTY index for WIDTH
cap.set(4, 720) # 4 - PROPERTY index for HEIGHT

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors','thanos','snake'])
            winner = calculate_winner(user_move_name, computer_move_name)

        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400, 400))
        frame[100:500, 800:1200] = icon

    cv2.imshow("AI Game", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyWindow()
cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
