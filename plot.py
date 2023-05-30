from matplotlib import pyplot as plt

# Suppose the given data strings are data1, data2, ..., data6
data1 = """
Epoch 1, Validation Loss: 1112101.0842317282
Epoch 2, Validation Loss: 719459.4456721559
Epoch 3, Validation Loss: 538648.3276170747
Epoch 4, Validation Loss: 506825.25704077206
Epoch 5, Validation Loss: 394801.9010578574
Epoch 6, Validation Loss: 353163.5705616834
Epoch 7, Validation Loss: 346141.84983545745
Epoch 8, Validation Loss: 302189.71886329434
Epoch 9, Validation Loss: 324146.00469935016
Epoch 10, Validation Loss: 328851.2783269472
Epoch 11, Validation Loss: 322268.86401912704
Epoch 12, Validation Loss: 264814.7880968605
Epoch 13, Validation Loss: 323717.6403576773
Epoch 14, Validation Loss: 396496.7335491038
Epoch 15, Validation Loss: 273452.96557028225
Epoch 16, Validation Loss: 330579.3209189619
Epoch 17, Validation Loss: 501321.24927362666
Epoch 18, Validation Loss: 550645.8989096085
Epoch 19, Validation Loss: 356358.3583924572
Epoch 20, Validation Loss: 303201.80416226876
Epoch 21, Validation Loss: 295392.65266450436
Epoch 22, Validation Loss: 276591.03075267834
Epoch 23, Validation Loss: 257184.7780770624
Epoch 24, Validation Loss: 299664.3431384668
Epoch 25, Validation Loss: 256635.15572342594
Epoch 26, Validation Loss: 227213.31567146344
Epoch 27, Validation Loss: 265269.98642823455
Epoch 28, Validation Loss: 201616.7909415593
Epoch 29, Validation Loss: 212601.08140528895
Epoch 30, Validation Loss: 219513.89173317555"""
data2 = """
Epoch 1, Validation Loss: 1068614.7496229103
Epoch 2, Validation Loss: 630526.017917182
Epoch 3, Validation Loss: 471318.5069773023
Epoch 4, Validation Loss: 386285.42429359304
Epoch 5, Validation Loss: 342438.95227487874
Epoch 6, Validation Loss: 307649.77981084905
Epoch 7, Validation Loss: 279437.3632256429
Epoch 8, Validation Loss: 262020.06927101934
Epoch 9, Validation Loss: 230864.25246383052
Epoch 10, Validation Loss: 212732.95617764496
Epoch 11, Validation Loss: 272415.0592748785
Epoch 12, Validation Loss: 209424.47819671044
Epoch 13, Validation Loss: 198904.31725547693
Epoch 14, Validation Loss: 238949.11748093105
Epoch 15, Validation Loss: 181628.49279202343
Epoch 16, Validation Loss: 190122.5059371525
Epoch 17, Validation Loss: 178670.6603817624
Epoch 18, Validation Loss: 155444.79288817875
Epoch 19, Validation Loss: 250591.8812130847
Epoch 20, Validation Loss: 187356.46489960985
Epoch 21, Validation Loss: 184704.89086306095
Epoch 22, Validation Loss: 198908.87932050604
Epoch 23, Validation Loss: 189757.6684171522
Epoch 24, Validation Loss: 178322.01159322186
Epoch 25, Validation Loss: 174688.4104154035
Epoch 26, Validation Loss: 230003.44203234956
Epoch 27, Validation Loss: 172532.03719271204
Epoch 28, Validation Loss: 212462.28757444944
Epoch 29, Validation Loss: 198608.55170161207
Epoch 30, Validation Loss: 189314.0677487972"""
data3 = """
Epoch 1, Validation Loss: 1154072.2291340968
Epoch 2, Validation Loss: 691684.178246335
Epoch 3, Validation Loss: 475879.87758454983
Epoch 4, Validation Loss: 396345.53189665556
Epoch 5, Validation Loss: 322840.48409273743
Epoch 6, Validation Loss: 293984.614943542
Epoch 7, Validation Loss: 258572.74423542302
Epoch 8, Validation Loss: 251690.59911739826
Epoch 9, Validation Loss: 224378.5199593798
Epoch 10, Validation Loss: 213667.5119355469
Epoch 11, Validation Loss: 215117.71072323938
Epoch 12, Validation Loss: 188463.20162284368
Epoch 13, Validation Loss: 181611.8358650676
Epoch 14, Validation Loss: 168888.77335172
Epoch 15, Validation Loss: 168055.87457948693
Epoch 16, Validation Loss: 187669.06634128993
Epoch 17, Validation Loss: 172789.44401696642
Epoch 18, Validation Loss: 180516.92892599257
Epoch 19, Validation Loss: 148604.8519039979
Epoch 20, Validation Loss: 150653.9408209041
Epoch 21, Validation Loss: 131378.06736827496
Epoch 22, Validation Loss: 136909.75076852055
Epoch 23, Validation Loss: 172951.25419386002
Epoch 24, Validation Loss: 132896.6761859588
Epoch 25, Validation Loss: 130161.40630758755
Epoch 26, Validation Loss: 116359.30619526585
Epoch 27, Validation Loss: 112751.76595082523
Epoch 28, Validation Loss: 121335.15329494416
Epoch 29, Validation Loss: 142761.69869704742
Epoch 30, Validation Loss: 129768.7543020641"""
data4 = """
Epoch 1, Validation Loss: 1101863.1034877584
Epoch 2, Validation Loss: 712693.7484476453
Epoch 3, Validation Loss: 543179.7868751318
Epoch 4, Validation Loss: 464433.1982310347
Epoch 5, Validation Loss: 399435.876382851
Epoch 6, Validation Loss: 365741.14755962125
Epoch 7, Validation Loss: 334788.3300091223
Epoch 8, Validation Loss: 315646.42653347
Epoch 9, Validation Loss: 288764.9326135782
Epoch 10, Validation Loss: 296399.6616173645
Epoch 11, Validation Loss: 266274.96979788376
Epoch 12, Validation Loss: 287832.1242105421
Epoch 13, Validation Loss: 243455.32231170352
Epoch 14, Validation Loss: 257248.04324930906
Epoch 15, Validation Loss: 258442.86567449945
Epoch 16, Validation Loss: 246875.86850911024
Epoch 17, Validation Loss: 241208.49869868628
Epoch 18, Validation Loss: 242634.9527867785
Epoch 19, Validation Loss: 243849.4240000589
Epoch 20, Validation Loss: 215716.64719060945
Epoch 21, Validation Loss: 221713.9525509979
Epoch 22, Validation Loss: 221838.91679974398
Epoch 23, Validation Loss: 236323.75914224927
Epoch 24, Validation Loss: 386485.04631082964
Epoch 25, Validation Loss: 1088122.173348178
Epoch 26, Validation Loss: 453045.87059442315
Epoch 27, Validation Loss: 337506.4944254169
Epoch 28, Validation Loss: 293437.54954929755
Epoch 29, Validation Loss: 244115.51850831095
Epoch 30, Validation Loss: 227101.13448062522
Epoch 31, Validation Loss: 204785.02542152654
Epoch 32, Validation Loss: 225693.5968540052
Epoch 33, Validation Loss: 221268.3001355379
Epoch 34, Validation Loss: 268420.4250903752
Epoch 35, Validation Loss: 216293.24550082625
Epoch 36, Validation Loss: 202890.26277332785
Epoch 37, Validation Loss: 199654.65587357924
Epoch 38, Validation Loss: 336053.3962765407
Epoch 39, Validation Loss: 208462.10218264096
Epoch 40, Validation Loss: 191259.1743042649"""
data5 = """
Epoch 1, Validation Loss: 1074024.910062742
Epoch 2, Validation Loss: 647403.5829355681
Epoch 3, Validation Loss: 488727.46757832414
Epoch 4, Validation Loss: 407384.21034571924
Epoch 5, Validation Loss: 348860.911646633
Epoch 6, Validation Loss: 310073.0863467725
Epoch 7, Validation Loss: 289566.5205223124
Epoch 8, Validation Loss: 273472.6567806718
Epoch 9, Validation Loss: 256692.97251467567
Epoch 10, Validation Loss: 226570.66618887967
Epoch 11, Validation Loss: 211734.2238545845
Epoch 12, Validation Loss: 205669.66540298532
Epoch 13, Validation Loss: 214958.79239944843
Epoch 14, Validation Loss: 223333.57828924022
Epoch 15, Validation Loss: 190789.84266543612
Epoch 16, Validation Loss: 207241.88694232103
Epoch 17, Validation Loss: 203183.05157810636
Epoch 18, Validation Loss: 188474.7649591254
Epoch 19, Validation Loss: 179619.3111251559
Epoch 20, Validation Loss: 178687.05000429333
Epoch 21, Validation Loss: 184831.1841580043
Epoch 22, Validation Loss: 170281.21296648314
Epoch 23, Validation Loss: 198584.95528276087
Epoch 24, Validation Loss: 159146.61687479343
Epoch 25, Validation Loss: 186760.45857714408
Epoch 26, Validation Loss: 190179.20257034092
Epoch 27, Validation Loss: 187832.25149614405
Epoch 28, Validation Loss: 188324.31208131148
Epoch 29, Validation Loss: 204284.29221783092
Epoch 30, Validation Loss: 186155.9760037683
Epoch 31, Validation Loss: 156667.0962568754
Epoch 32, Validation Loss: 212298.71865440538
Epoch 33, Validation Loss: 221196.16497445107
Epoch 34, Validation Loss: 213876.01451977543
Epoch 35, Validation Loss: 195191.45390687228
Epoch 36, Validation Loss: 173767.51286355016
Epoch 37, Validation Loss: 137705.7925829149
Epoch 38, Validation Loss: 160849.10911054618
Epoch 39, Validation Loss: 165724.3406425793
Epoch 40, Validation Loss: 241905.20881354623"""
data6 = """
Epoch 1, Validation Loss: 1159557.2247891747
Epoch 2, Validation Loss: 655307.8731871947
Epoch 3, Validation Loss: 476162.1703109415
Epoch 4, Validation Loss: 394740.8585001562
Epoch 5, Validation Loss: 334483.2309794739
Epoch 6, Validation Loss: 301044.08134956384
Epoch 7, Validation Loss: 266651.2639557961
Epoch 8, Validation Loss: 257547.20915961865
Epoch 9, Validation Loss: 243835.77208197117
Epoch 10, Validation Loss: 249558.6638125779
Epoch 11, Validation Loss: 209833.65018178232
Epoch 12, Validation Loss: 196303.51771599392
Epoch 13, Validation Loss: 183482.1958622895
Epoch 14, Validation Loss: 171815.7128740403
Epoch 15, Validation Loss: 177703.8200525885
Epoch 16, Validation Loss: 172615.79236469217
Epoch 17, Validation Loss: 160447.503295714
Epoch 18, Validation Loss: 159414.22163029885
Epoch 19, Validation Loss: 163243.117377842
Epoch 20, Validation Loss: 149584.90758724182
Epoch 21, Validation Loss: 150360.79658584774
Epoch 22, Validation Loss: 138085.86878239358
Epoch 23, Validation Loss: 130357.44285402958
Epoch 24, Validation Loss: 145010.2737539969
Epoch 25, Validation Loss: 159746.0224329773
Epoch 26, Validation Loss: 138792.22467461322
Epoch 27, Validation Loss: 142933.06180246023
Epoch 28, Validation Loss: 141830.39314052294
Epoch 29, Validation Loss: 122964.71697767575
Epoch 30, Validation Loss: 117822.95883853237
Epoch 31, Validation Loss: 147530.21277802845
Epoch 32, Validation Loss: 130453.66691763896
Epoch 33, Validation Loss: 120272.82311328551
Epoch 34, Validation Loss: 119016.98443477409
Epoch 35, Validation Loss: 170541.9924689509
Epoch 36, Validation Loss: 126389.16168100054
Epoch 37, Validation Loss: 117673.68080671965
Epoch 38, Validation Loss: 128790.82941057248
Epoch 39, Validation Loss: 122563.8897286811
Epoch 40, Validation Loss: 124684.02987675097"""

# Create a list of data strings and corresponding labels
data_list = [data1, data2, data3, data4, data5, data6]
labels = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6"]

# Function to extract losses from data string
def extract_losses(data_string):
    return [float(line.split(": ")[1]) for line in data_string.split("\n") if line]

# Extract losses for each data string
losses_list = [extract_losses(data) for data in data_list]

# Plot losses for each model
plt.figure(figsize=(10, 6))
for i, losses in enumerate(losses_list):
    plt.plot(range(1, len(losses) + 1), losses, label=labels[i])

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Training and Evaluation loss for LSTM models')
plt.legend()
plt.grid(True)
plt.show()
