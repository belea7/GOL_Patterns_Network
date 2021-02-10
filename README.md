Game of Life cyclic patterns recognition
--------------
**This program implements the ANN from scratch (without using external modules).**

**Python modules: pyseagle, matplotlib, numpy.**

**Final project at Biological Computation course at The Open University of Israel**

Game of Life if a popular cellular automaton: https://playgameoflife.com/
It contains interesting patterns that behave in certain ways, for example cyclic patterns that return to their initial state after certain amount of generations. For example:


![alt text](https://www.conwaylife.com/w/images/b/b7/Dinnertable.gif?raw=true)


This wiki contains all the discovered interesting patterns: https://www.conwaylife.com/wiki/Main_Page

I created a neural network that receives a 10x10 Game of Life grid and classifies it as cyclic or not cyclic. Cyclic patterns are patterns that return to their initial state after some generations. The network has 3 layers: input layer with 100 cells, hidden layer with 5 neurons and output layer with 2 neurons. The network is trained with backprop algorithm that I fully implemented and reaches 80% success rate.
