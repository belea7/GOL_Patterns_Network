Game of Life cyclic patterns recognition
--------------
**This program implements the ANN from scratch (without using external modules).**

**Python modules: pyseagle, matplotlib, numpy.**

**Final project at Biological Computation course at The Open University of Israel**

I created a neural network that receives a 10x10 Game of Life grid and classifies it as cyclic or not cyclic. Cyclic patterns are patterns that return to their initial state after some generations. The network has 3 layers: an input layer with 100 cells, a hidden layer with 5 neurons, and an output layer with 2 neurons. The network is trained with a backdrop algorithm that I fully implemented and reaches an 80% success rate.

Game of Life if a popular cellular automaton: https://playgameoflife.com/

It contains interesting patterns with complex behavior, for example cyclic patterns that return to their initial state after certain amount of generations. For example:


![alt text](https://www.conwaylife.com/w/images/b/b7/Dinnertable.gif?raw=true)


This wiki contains all the discovered interesting patterns: https://www.conwaylife.com/wiki/Main_Page
