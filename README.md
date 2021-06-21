# Fast PDN Impedance Prediction Using Deep Learning
This Python code is the algorithm for the paper: Fast PDN Impedance Prediction Using Deep Learning, which has been sumbitted to **International Journal of Numerical Modeling: Electronic Networks, Devices and Fields.**

Author: Ling Zhang; email: lingzhang_zju@zju.edu.cn

`L. Zhang, J. Juang, Z. Kiguradze, B. Pu, S. Jin, S. Wu, Z. Yang, and C. Hwang, “Fast PDN Impedance Prediction Using Deep Learning,” International Journal of Numerical Modeling: Electronic Networks, Devices and Fields, submitted.`

Abstract: Modeling and simulating a power distribution network (PDN) for printed circuit boards (PCBs) with irregular shapes and multi-layer stackups is computationally inefficient using full-wave simulations. This paper presents a new concept of using deep learning for PDN impedance prediction. A boundary element method (BEM) is applied to efficiently calculate the impedance for arbitrary board shape and stackup. Then, over one million boards with different shapes, stackup, IC location, and decap placement are randomly generated to train a deep neural network (DNN). The trained DNN can predict the impedance accurately for new board configurations that have not been used for training. The consumed time using the trained DNN is only 0.1 seconds, which is over 100 times faster than the BEM method and 5000 times faster than full-wave simulations.


The paper can also be found on [arXiv]().

## Code Usage
- 1. Run **_gen_brd.py_** to generate PCB boards with different shapes(the maximum area is 200mm by 200mm), stackups(number of layers is 4~9), and candidate decap locations (19 locations maximum, distance between power to ground via is 2mm).
- 2. Run **_gen_supervised_data.py_** to generate data for supervised learning using the board data generated in step i (now for each board generated in step 1, decap number ranges from 0 to 19, and the locations are random). 
- 3. Run **_train_supervised.py_** to train the supervised learning algorithm to predict PDN impedance.
- 4. Run **_read_supervised_model.py_** to view the prediction result using the trained model in step iii.

## gen_brd.py
Define a PDN class, which is defined in pdn_class.py
```
  brd = PDN()
```
Generate a board with a random board shape, random stackup, and random decap locations.
The maximum board size is 200mm by 200mm. The plane is discretized into a 16 by 16 matrix. 
```
  z, brd_shape_ic, ic_xy_indx, top_decap_xy_indx, bot_decap_xy_indx, vrm_xy_indx, vrm_loc, stackup, die_t, t, d2top, sxy, ic_via_xy, ic_via_type, decap_via_xy, decap_via_type, decap_via_loc, max_len = gen_brd_data(brd=brd, max_len=200e-3, img_size=16)
```
Function to generate a random board shape, which is borrowed from [this website.](https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib)
```
  def gen_random_brd_shape(max_len, img_size, num_locations, n=8, thre_angle=30, 
                           rad=0.1, edgy=0.05, num_pts=10):
```
Function to generate a random stackup:
```
  def gen_random_stackup(min_t, max_t, min_dt, min_n_layer, max_n_layer):
```
Function to generate a board with a random board shape, random stackup, and random decap locations.
- max_len: the maximum board size (200mm is used)
- img_size: how many pixels are used for the matrix (16 is used)
- num_locations: how many random locations are generated (20 is used, IC location is also included)
- min_t: the minimum board thickness (1mm by default)
- max_t: the maximum board thickness (5mm by default)
- min_n_layer: the minimum number of layers (4 by default)
- max_n_layer: the maximum number of layers (9 by default)
- via_dist: the distance between power and ground vias for decaps and IC (2mm by default. Maybe too large for a real board design)

```
  def gen_brd_data(brd, max_len, img_size, num_locations=20, n=8, thre_angle=30, rad=0.1, edgy=0.05, 
                   num_pts=10, min_t=1e-3, max_t=10e-3, min_dt=0.1e-3, min_n_layer=4, max_n_layer=9, 
                   er=4.4, via_dist=2e-3, vrm_r=3e-3, vrm_l=2.5e-9):
```

## gen_supervised_data.py
Define the path to read the board data, as well as the path to store the supervised data
```
  BASE_PATH = 'brd_data/'
  NEW_DATA_PATH = 'supervised_data/'
```
Define the repetition factor, which means how many repetitive cases (decap number is the same, but decap locations and decap models are different) are generated for each scenario.
```
  repeat = 5
```
Specify the board files to be read:
```
  file_list = list(range(0,10000))
```
Read board information from the board files:
```
  z_orig = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['z']
  brd_shape_ic = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['brd_shape_ic']
  ic_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['ic_xy_indx']
  top_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['top_decap_xy_indx']
  bot_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['bot_decap_xy_indx']
  stackup = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['stackup']
  die_t = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['die_t']
  sxy = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['sxy']
```
Connect the board Z-parameters with decap Z-paramters:
```
  z, map2orig_output = connect_1decap(z, map2orig_output, 
                                      map2orig_output.index(j), 
                                      brd.decap_list[decap_indx])
```
Save the Z-parameter files with connected decaps:
```
  np.savez(os.path.join(NEW_DATA_PATH, str(n*(z_orig.shape[1]*repeat)+r*z_orig.shape[1]+i)+'.npz'), 
                       x1=x1, x2=x2, y=y, sxy=sxy)
```

## train_supervised.py
The following DNN structure is used:
![fig8](https://user-images.githubusercontent.com/33564605/122697528-1a0e8600-d278-11eb-9c41-bc02302d86c1.png)

## read_supervised_model.py
Specify the path of the trained model:
```
  dataname = 'test3_no_normalization'
  NEW_DATA_PATH = 'supervised_data/'
  SUMMARY_PATH = "source/summary/" + dataname
  MODEL_PATH = 'trained_model/'
```
The trained model is loaded here:
```
  net.load_state_dict(torch.load(os.path.join(MODEL_PATH, dataname +'_best_model'), map_location=torch.device('cpu')))
```
Randomly select a test case:
```
  # ID = random.randint(0, 950000-10000)
  ID = random.randint(950000-10000, 950000)   # randomly select a case from the testing data
  # ID = 941762
```

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

