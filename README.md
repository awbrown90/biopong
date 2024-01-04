BioPong: Biologically Iinspired Machine Learning

![Animation](images/biopong_rallies.gif)

BioPong is a machine learning research project inspired by recent studies where biological brain organoids played Pong from inside a laboratory petri dish. What was most interesting about the biological experiment was the observation that the neurons' behavior could be controlled by their desire to maximize expectation and minimize surprise.
The cells could be motivated to make the paddle hit the ball by introducing random input signals, which are impossible to predict, for a brief duration whenever the paddle missed the ball.  [Read more about the inspiration for this project.](https://neurosciencenews.com/organoid-pong-21625/)

This biological learning approach is actually very different from traditional methods used in deep reinforcement learning, which, for the most part, rely on the Bellman Equation and policy fitting with known rewards. 
The BioPong research project aimed to determine whether computer neural networks could learn to play pong under the same conditions as biological cells, without any direct rewards and using only random signals as a negative feedback indicator.
