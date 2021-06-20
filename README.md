# Fast PDN Impedance Prediction Using Deep Learning
This Python code is the algorithm for the paper: Fast PDN Impedance Prediction Using Deep Learning, which has been sumbitted to International Journal of Numerical Modeling: Electronic Networks, Devices and Fields.

***Abstract: Modeling and simulating a power distribution network (PDN) for printed circuit boards (PCBs) with irregular shapes and multi-layer stackups is computationally inefficient using full-wave simulations. This paper presents a new concept of using deep learning for PDN impedance prediction. A boundary element method (BEM) is applied to efficiently calculate the impedance for arbitrary board shape and stackup. Then, over one million boards with different shapes, stackup, IC location, and decap placement are randomly generated to train a deep neural network (DNN). The trained DNN can predict the impedance accurately for new board configurations that have not been used for training. The consumed time using the trained DNN is only 0.1 seconds, which is over 100 times faster than the BEM method and 5000 times faster than full-wave simulations.***


The paper can also be found on arXiv: 

## Code Explanation
- 1. Run gen_brd.py to generate PCB boards with different shapes, stackups, and candidate decap locations (19 locations maximum).
- 2. Run gen_supervised_data.py to generate data for supervised learning using the board data generated in step i. 
- 3. Run train_supervised.py to train the supervised learning algorithm to predict PDN impedance.
- 4. Run read_supervised_model.py to view the prediction result using the trained model in step iii.

## Training process
![fig9](https://user-images.githubusercontent.com/33564605/122676465-d3d20c00-d210-11eb-9016-7033c914a353.png)

## Prediction result using trained model
### Randomly selected case 1
![fig10a](https://user-images.githubusercontent.com/33564605/122676504-f8c67f00-d210-11eb-8ff8-411e198de4eb.png)
![fig10b](https://user-images.githubusercontent.com/33564605/122676506-fb28d900-d210-11eb-8c13-18cb43768e62.png)
![image](https://user-images.githubusercontent.com/33564605/122676526-13005d00-d211-11eb-8aa1-f5728b8ed0f0.png)

### Randomly selected case 2
![fig11a](https://user-images.githubusercontent.com/33564605/122676579-3cb98400-d211-11eb-8d49-a4fb63d3d6d2.png)
![fig11b](https://user-images.githubusercontent.com/33564605/122676580-3e834780-d211-11eb-99c9-784052ce0864.png)
![image](https://user-images.githubusercontent.com/33564605/122676592-4c38cd00-d211-11eb-8664-fc47c3de7d0e.png)

