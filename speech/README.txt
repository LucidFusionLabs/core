Training Models With LFL Coder
------------------------------

Input
-----

dir \corpus\wav\voxforge

06/08/2009  06:51 AM           846,955 anonymous-20090603-lzd.tgz
06/08/2009  06:52 AM           667,004 anonymous-20090605-dlc.tgz
06/08/2009  06:51 AM         1,002,472 anonymous-20090605-iyu.tgz



Initial Model
-------------

coder -T -I 39 -M 1 -m 1 -i 3         # init 1 comp model

dir \corpus\model\*

11/03/2009  07:35 PM               483 Model_AH_State_02.0000.emCov.matrix
11/03/2009  07:35 PM               485 Model_AH_State_02.0000.emMeans.matrix
11/03/2009  07:35 PM               142 Model_AH_State_02.0000.emPrior.matrix
11/03/2009  07:35 PM               182 Model_AH_State_02.0000.transition.matrix
11/03/2009  07:35 PM               483 Model_AO_State_00.0000.emCov.matrix
11/03/2009  07:35 PM               485 Model_AO_State_00.0000.emMeans.matrix

head Model_AH_State_02.0001.emMeans.matrix

Model_AH_State_02.0001.emMeans.matrix
sr=16000,fftlen=512,ffthop=256,mels=40,ceps=13,zeroth,cmn,cvn,delta,deltadelta
Matrix
32 39
-7.135417 16.392056 -1.853003 1.464456 -3.486269 0.001893 -2.136094 -0.510008 ...



Features
--------

coder -F                              # make features

dir \corpus\features\*

11/03/2009  01:16 PM     1,335,270,626 voxforge.featlist

head voxforge.featlist

Aaron-20080318-kdl/wav/b0022.wav
IF I WAS OUT OF THE GAME IT WOULD BE EASILY MADE
Matrix
249 13
-123.646213 5.356663 -0.099115 3.868976 4.705149 2.007924 0.829505 2.836598 ...



Training
--------

coder -T -I 39 -M 1 -m 1 -i 3                          # init 1 comp model
coder -T                                               # train
dir \corpus\model\*

coder -T -P initial.pathlist                           # viterbi paths
del \corpus\model\*

coder -T -I 39 -M 4 -p initial.pathlist -m 1 -i 5      # init 4 comp model from paths
coder -T                                               # train
coder -T -P four.pathlist -m 1 -i 0                    # save final paths
del \corpus\model\*

coder -T -I 39 -M 32 -p four.pathlist -m 1 -i 5        # cluster 32 comp from paths
coder -T                                               # train



Decoding
--------

coder -d \goo.wav


XXX Needs Update
----------------

coder -feats2model 1 -Initialize 60 -MixtureComponents 12 -MaxIterations 1 -MeansIterations 3 -UttPathsInName vp.0012.matlist



