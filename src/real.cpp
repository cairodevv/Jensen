#include <iostream>
#include <cstdlib>
#include <utility>

std::pair<int, int> guessWithMultiply(int number, int changeDifference) {
    int tempChangeDifference = {changeDifference};
    bool finished = {false};
    int guess = {rand() % 10001};
    int loss = {abs(number - (guess*8)+6)};
    bool directionUp = {(rand() % 101) >= 50};
    int prevLoss[2] = {100, 100};
    int repeatCount = {0};
    int i = {0};
    while (!finished) {
        if (loss == 0) {
            finished = {true};
            continue;
        }
        if (loss == prevLoss[0]) {
            if (repeatCount < 5) {
                repeatCount++;
            } else {
                finished = {true};
                continue;
            }
        }
        if (loss > prevLoss[1]) {
            directionUp = !directionUp;
            tempChangeDifference = changeDifference;
            i = {0};
        }
        prevLoss[0] = prevLoss[1];
        prevLoss[1] = loss;
        if (directionUp) {
            if (guess + tempChangeDifference >= 10000) {
                guess = {10000};
            } else {
                guess += tempChangeDifference;
            }
        } else {
            if (guess - tempChangeDifference <= 0) {
                guess = {0};
            } else {
                guess -= tempChangeDifference;
            }
        }
        loss = {abs(number - (guess*8) + 6)};
        if (i == 10) {
            tempChangeDifference *= 2;
            i = {0};
        } else {
            i++;
        }
    }
    return std::pair(guess, loss);
}
int main() {
    int number;
    int diff;
    std::cin >> number;
    std::cin >> diff;
    std::pair<int, int> guess = guessWithMultiply(number, diff);
    std::cout << "Guess: " << guess.first << "\nLoss: " << guess.second << "\n";
    return 0;
}