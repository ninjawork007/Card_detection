# Import necessary packages
import cv2
import os
import Cards1
# import matplotlib.pyplot as plt

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX
# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards1.load_ranks(path + '/Card_Imgs/')
train_suits = Cards1.load_suits(path + '/Card_Imgs/')

### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0  # Loop control variable

# Begin capturing frames
image = cv2.imread("baccarat-cards/8C_7H_5H_7H_6S_2C.png")
# Start timer (for calculating frame rate)
t1 = cv2.getTickCount()

# Pre-process camera image (gray, blur, and threshold it)
pre_proc = Cards1.preprocess_image(image)
# plt.imshow(pre_proc)
# plt.show()

CountSpades = 0
CountHearts = 0
CountClubs = 0
CountDiamonds = 0
TotalRank = 0

# global variable array
global cards
cards = []
# Find and sort the contours of all cards in the image (query cards)
image1=image.copy()
cnts_sort, cnt_is_card = Cards1.find_cards(pre_proc)

# If there are no contours, do nothing
if len(cnts_sort) != 0:

    # Initialize a new "cards" list to assign the card objects.
    # k indexes the newly made array of cards.
    # cards = []
    k = 0

    # For each contour detected:
    for i in range(len(cnts_sort)):
        if (cnt_is_card[i] == 1):
            # Create a card object from the contour and append it to the list of cards.
            # preprocess_card function takes the card contour and contour and
            # determines the cards properties (corner points, etc). It generates a
            # flattened 200x300 image of the card, and isolates the card's
            # suit and rank from the image.
            cards.append(Cards1.preprocess_card(cnts_sort[i], image1))

            # Find the best rank and suit match for the card.
            cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[
                k].suit_diff = Cards1.match_card(cards[k], train_ranks, train_suits)

            # Draw center point and match result on the image.
            # image = Cards1.draw_results(image, cards[k])
            k = k + 1

    # Draw card contours on image (have to do contours all at once or
    # they do not show up properly for some reason)
    if (len(cards) != 0):
        temp_cnts = []

        for i in range(len(cards)):

            if cards[i].best_rank_match in ['Ace']:
                TotalRank = TotalRank + 1

            if cards[i].best_rank_match in ['Two']:
                TotalRank = TotalRank + 2

            if cards[i].best_rank_match in ['Three']:
                TotalRank = TotalRank + 3

            if cards[i].best_rank_match in ['Four']:
                TotalRank = TotalRank + 4

            if cards[i].best_rank_match in ['Five']:
                TotalRank = TotalRank + 5

            if cards[i].best_rank_match in ['Six']:
                TotalRank = TotalRank + 6

            if cards[i].best_rank_match in ['Seven']:
                TotalRank = TotalRank + 7

            if cards[i].best_rank_match in ['Eight']:
                TotalRank = TotalRank + 8

            if cards[i].best_rank_match in ['Nine']:
                TotalRank = TotalRank + 9

            if cards[i].best_rank_match in ['Ten']:
                TotalRank = TotalRank + 10

            if cards[i].best_rank_match in ['Jack']:
                TotalRank = TotalRank + 11

            if cards[i].best_rank_match in ['Queen']:
                TotalRank = TotalRank + 12

            if cards[i].best_rank_match in ['King']:
                TotalRank = TotalRank + 13

            if cards[i].best_suit_match in ['Spades']:
                CountSpades += 1

            if cards[i].best_suit_match in ['Hearts']:
                CountHearts += 1

            if cards[i].best_suit_match in ['Clubs']:
                CountClubs += 1

            if cards[i].best_suit_match in ['Diamonds']:
                CountDiamonds += 1

            temp_cnts.append(cards[i].contour)
        cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

# Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
# so the first time this runs, framerate will be shown as 0.
cv2.putText(image, "Total Cards " + str(len(cards)), (10, 26), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(image, "Spades: " + str(int(CountSpades)), (10, 56), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(image, "Hearts: " + str(int(CountHearts)), (10, 86), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(image, "Clubs: " + str(int(CountClubs)), (10, 116), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(image, "Diamonds: " + str(int(CountDiamonds)), (10, 146), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(image, "Total of Ranks: " + str(int(TotalRank)), (10, 176), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# Finally, display the image with the identified cards!
cv2.imshow("Card Detector", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

