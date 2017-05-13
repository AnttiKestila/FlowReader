# coding=utf-8
import matplotlib.pyplot as plt
from numpy import loadtxt, average, arange, polyfit, poly1d
from Readertools import Reader

__author__ = 'Antti Kestilä 2015'

"""
data = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi1000ABSP-1_130916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data200 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi1000ABSP-2_130916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data1000 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi1000ABSP-1_150916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data2000 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi1000ABSP-1_160916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

data = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSP-1_150916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data200 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnew-1_160916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data1000 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnew-4_200916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data2000 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnew-3_220916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

data = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnewnew-1_031016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data200 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnewnew-2_031016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data1000 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnewnew-3_031016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data2000 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnewnew-4_031016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

dataPA = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnewnew-1_041016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA200 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnewnew-2_041016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA1000 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi0ABSPnewnew-3_041016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA2000 = loadtxt("C:/Users/Antti/Downloads/yunBackup/venturi2000PAnew-4_230916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

data1 = loadtxt("/home/antti/Downloads/yunBackup/venturi0ABSPnewnewnew-1_051016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data2 = loadtxt("/home/antti/Downloads/yunBackup/venturi0ABSPnewnewnew-2_051016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data3 = loadtxt("/home/antti/Downloads/yunBackup/venturi0ABSPnewnewnew-3_051016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data4 = loadtxt("/home/antti/Downloads/yunBackup/venturi0ABSPnewnewnew-4_051016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data5 = loadtxt("/home/antti/Downloads/yunBackup/venturi0ABSPnewnewnew-5_051016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data6 = loadtxt("/home/antti/Downloads/yunBackup/venturi0ABSPnewnewnew-6_051016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

data1 = loadtxt("/home/antti/Downloads/yunBackup/venturi1000ABSPnew-1_220916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data2 = loadtxt("/home/antti/Downloads/yunBackup/venturi1000ABSPnew-2_220916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data3 = loadtxt("/home/antti/Downloads/yunBackup/venturi1000ABSPnew-3_220916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data4 = loadtxt("/home/antti/Downloads/yunBackup/venturi1000ABSPnew-4_220916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data5 = loadtxt("/home/antti/Downloads/yunBackup/venturi1000ABSPnew-3_220916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data6 = loadtxt("/home/antti/Downloads/yunBackup/venturi1000ABSPnew-4_220916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

data1 = loadtxt("/home/antti/Downloads/yunBackup/venturi2000ABSPlus-1_070916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data2 = loadtxt("/home/antti/Downloads/yunBackup/venturi2000ABSP-2_140916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data3 = loadtxt("/home/antti/Downloads/yunBackup/venturi2000ABSP-1_190916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data4 = loadtxt("/home/antti/Downloads/yunBackup/venturi2000ABSP-2_190916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data5 = loadtxt("/home/antti/Downloads/yunBackup/venturi2000ABSP-3_190916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data6 = loadtxt("/home/antti/Downloads/yunBackup/venturi2000ABSP-4_190916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

dataPA1 = loadtxt("/home/antti/Downloads/yunBackup/venturi200ABSP-1_130916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA2 = loadtxt("/home/antti/Downloads/yunBackup/venturi200ABSP-2_130916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA3 = loadtxt("/home/antti/Downloads/yunBackup/venturi200ABSP-1_150916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA4 = loadtxt("/home/antti/Downloads/yunBackup/venturi200ABSP-2_150916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA5 = loadtxt("/home/antti/Downloads/yunBackup/venturi200ABSP-3_210916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA6 = loadtxt("/home/antti/Downloads/yunBackup/venturi200ABSP-4_210916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

"""

data1 = loadtxt("/home/antti/Downloads/yunBackup/venturi0ABSPnewnewnew-6_051016.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data2 = loadtxt("/home/antti/Downloads/yunBackup/venturi200ABSP-4_210916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data3 = loadtxt("/home/antti/Downloads/yunBackup/venturi1000ABSPnew-4_220916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
data4 = loadtxt("/home/antti/Downloads/yunBackup/venturi2000ABSP-4_190916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

dataPA1 = loadtxt("/home/antti/Downloads/yunBackup/venturi0PA-1_070916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA2 = loadtxt("/home/antti/Downloads/yunBackup/venturi200PA-1_080916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA3 = loadtxt("/home/antti/Downloads/yunBackup/venturi1000PA-1_080916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})
dataPA4 = loadtxt("/home/antti/Downloads/yunBackup/venturi2000PAnew-4_230916.txt", delimiter=',', usecols={2, 3, 4, 5, 8, 12, 13, 16})

dataC = loadtxt("/home/antti/Downloads/yunBackup/hydacCalibration_020316-minP6.5e-3mbar.txt", delimiter=',', usecols={2, 3})

# Other local parameters
thrt = 0.00016
thrtPA = 0.00016
initialpressure = 1.01325
currenttemp = 32.6
# Argon interpolation and conversion from volumetric flow to mass flow
x_mf = [0.5, 0.9, 1.3, 1.7, 2.1, 2.5]
y_data_mf = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
thres = 0.2
md = 25
N = 1
calibration = 0.0
calibrationPA = calibration
swtch = True

data1 = Reader(data1, dataC)
p1c1, p2c1, massflow11, massflow21, m1cc1 = data1.plotter(initialpressure, thrt, x_mf, y_data_mf, currenttemp, calibration)
data2 = Reader(data2, dataC)
p1c2, p2c2, massflow12, massflow22, m1cc2 = data2.plotter(initialpressure, thrt, x_mf, y_data_mf, currenttemp, calibration)
data3 = Reader(data3, dataC)
p1c3, p2c3, massflow13, massflow23, m1cc3 = data3.plotter(initialpressure, thrt, x_mf, y_data_mf, currenttemp, calibration)
data4 = Reader(data4, dataC)
p1c4, p2c4, massflow14, massflow24, m1cc4 = data4.plotter(initialpressure, thrt, x_mf, y_data_mf, currenttemp, calibration)

if swtch:
    data5 = Reader(data5, dataC)
    p1c5, p2c5, massflow15, massflow25, m1cc5 = data5.plotter(initialpressure, thrt, x_mf, y_data_mf, currenttemp, calibration)
    data6 = Reader(data6, dataC)
    p1c6, p2c6, massflow16, massflow26, m1cc6 = data4.plotter(initialpressure, thrt, x_mf, y_data_mf, currenttemp, calibration)

data7 = Reader(dataPA1, dataC)
p1c7, p2c7, massflow17, massflow27, m1cc7 = data7.plotter(initialpressure, thrtPA, x_mf, y_data_mf, currenttemp, calibrationPA)
data8 = Reader(dataPA2, dataC)
p1c8, p2c8, massflow18, massflow28, m1cc8 = data8.plotter(initialpressure, thrtPA, x_mf, y_data_mf, currenttemp, calibrationPA)
data9 = Reader(dataPA3, dataC)
p1c9, p2c9, massflow19, massflow29, m1cc9 = data9.plotter(initialpressure, thrtPA, x_mf, y_data_mf, currenttemp, calibrationPA)
data10 = Reader(dataPA4, dataC)
p1c10, p2c10, massflow110, massflow210, m1cc10 = data10.plotter(initialpressure, thrtPA, x_mf, y_data_mf, currenttemp, calibrationPA)

if swtch:
    data11 = Reader(dataPA5, dataC)
    p1c11, p2c11, massflow111, massflow211, m1cc11 = data11.plotter(initialpressure, thrtPA, x_mf, y_data_mf, currenttemp, calibrationPA)
    data12 = Reader(dataPA6, dataC)
    p1c12, p2c12, massflow112, massflow212, m1cc12 = data12.plotter(initialpressure, thrtPA, x_mf, y_data_mf, currenttemp, calibrationPA)

cut = 2300


# Finding peaks for the actual flows
arrmf1, index1 = Reader.peaking(massflow11[cut:6000], thres, md)
arrmf2, index2 = Reader.peaking(massflow12[cut:6000], thres, md)
arrmf3, index3 = Reader.peaking(massflow13[cut:6000], thres, md)
arrmf4, index4 = Reader.peaking(massflow14[cut:6000], thres, md)

if swtch:
    arrmf5, index5 = Reader.peaking(massflow15[cut:6000], thres, md)
    arrmf6, index6 = Reader.peaking(massflow16[cut:6000], thres, md)

arrmf7, index7 = Reader.peaking(massflow17[cut:5000], thres, md)
arrmf8, index8 = Reader.peaking(massflow18[cut:5000], thres, md)
arrmf9, index9 = Reader.peaking(massflow19[cut:5000], thres, md)
arrmf10, index10 = Reader.peaking(massflow110[cut:5000], thres, md)

if swtch:
    arrmf11, index11 = Reader.peaking(massflow111[cut:5000], thres, md)
    arrmf12, index12 = Reader.peaking(massflow112[cut:5000], thres, md)


# Finding peaks for the flow differences
arrmf12, index12 = Reader.peaking(m1cc1[cut:6000] - massflow11[cut:6000], 0.15, 15) # md 10
arrmf22, index22 = Reader.peaking(m1cc2[cut:6300] - massflow12[cut:6300], 0.15, 15)
arrmf32, index32 = Reader.peaking(m1cc3[cut:5400] - massflow13[cut:5400], 0.08, 20)
arrmf42, index42 = Reader.peaking(m1cc4[cut:5500] - massflow14[cut:5500], 0.15, 20)

if swtch:
    arrmf52, index52 = Reader.peaking(m1cc5[cut:5700] - massflow15[cut:5700], 0.1, 20)# the last vector values is 5800
    arrmf62, index62 = Reader.peaking(m1cc6[cut:5560] - massflow16[cut:5560], 0.1, 19) #the last vector values is 5800

arrmf72, index72 = Reader.peaking(m1cc7[cut:6000] - massflow17[cut:6000], 0.6, 15) # the last vector values is 5800
arrmf82, index82 = Reader.peaking(m1cc8[cut:5300] - massflow18[cut:5300], 0.55, 15) # md 20 the last vector values is 5800
arrmf92, index92 = Reader.peaking(m1cc9[cut:5200] - massflow19[cut:5200], 0.45, 20)
arrmf102, index102 = Reader.peaking(m1cc10[cut:6000] - massflow110[cut:6000], 0.5, 15)

if swtch:
    arrmf112, index112 = Reader.peaking(m1cc11[cut:5900] - massflow111[cut:5900], 0.5, 15) # the last vector values is 5800
    arrmf122, index122 = Reader.peaking(m1cc12[cut:7000] - massflow112[cut:7000], 0.5, 15) # the last vector values is 5800

x1 = 3850
x2 = 4150
y1 = -0.5
y2 = 1.0
#print(average(arrmf12[index12]), average(arrmf22[index22]), average(arrmf32[index32]), average(arrmf42[index42]), average(arrmf52[index52]), average(arrmf62[index62]), average(arrmf72[index72]), average(arrmf82[index82]), average(arrmf92[index92]), average(arrmf102[index102]), average(arrmf112[index112]), average(arrmf122[index122]))

# Plotting everything

with plt.style.context('seaborn-whitegrid'):

    fig1 = plt.figure()

    ax = plt.subplot(2, 1, 1)
    plt.ylim((y1, y2))
    plt.xlim((x1, x2))
    plt.title(r'ABS Plus -$\Delta$mg/s')
    plt.ylabel('$\Delta$mg/s')
    plt.xlabel('Time units')

    plt.plot(m1cc1 - massflow11, 'b', label="Uncoated")
    plt.plot(m1cc2 - massflow12, 'g', label="20nm")
    plt.plot(m1cc3 - massflow13, 'r', label="100nm")
    plt.plot(m1cc4 - massflow14, 'c', label="200nm")

    #if swtch:
    #    plt.plot(m1cc5 - massflow15, 'y', label="rep.")
    #    plt.plot(m1cc6 - massflow16, 'k', label="rep.")

    plt.plot(index12 + cut, arrmf12[index12], 'r+')
    plt.plot(index22 + cut, arrmf22[index22], 'r+')
    plt.plot(index32 + cut, arrmf32[index32], 'b+')
    plt.plot(index42 + cut, arrmf42[index42], 'r+')

    #if swtch:
    #   plt.plot(index52 + cut, arrmf52[index52], 'r+')
    #    plt.plot(index62 + cut, arrmf62[index62], 'r+')

    plt.legend(loc="upper left", shadow=True, title="ABS Plus", frameon=True, bbox_to_anchor=[0.995, 1.0])

    ax = plt.subplot(2, 1, 2)

    plt.title('PA 2200 -$\Delta$mg/s')
    plt.ylabel('$\Delta$mg/s')
    plt.xlabel('Time units')
    plt.ylim((y1, y2))
    plt.xlim((x1, x2))
    plt.plot(m1cc7 - massflow17, 'b', label="Uncoated")
    plt.plot(m1cc8 - massflow18, 'g', label="20nm")
    plt.plot(m1cc9 - massflow19, 'r', label="100nm")
    plt.plot(m1cc10 - massflow110, 'c', label="200nm")

 #   if swtch:
    #    plt.plot(m1cc11 - massflow111, 'y', label="rep.")
    #    plt.plot(m1cc12 - massflow112, 'k', label="rep.")

    plt.plot(index72 + cut, arrmf72[index72], 'r+')
    plt.plot(index82 + cut, arrmf82[index82], 'r+')
    plt.plot(index92 + cut, arrmf92[index92], 'b+')
    plt.plot(index102 + cut, arrmf102[index102], 'r+')

#    if swtch:
    #    plt.plot(index112 + cut, arrmf112[index112], 'r+')
    #    plt.plot(index122 + cut, arrmf122[index122], 'r+')

    plt.legend(loc="upper left", shadow=True, title="PA 2200", frameon=True, bbox_to_anchor=[0.995, 1.0])
    plt.xlabel('time units')

"""
# second figure
fig2 = plt.figure()

ax2 = plt.subplot(2, 1, 1)
plt.plot(p1c2, 'b')

plt.plot(Reader.sum_range(massflow12, N), 'g')
plt.plot(Reader.sum_range(massflow13, N), 'r')
plt.plot(Reader.sum_range(massflow14, N), 'c')

if swtch:
    plt.plot(Reader.sum_range(massflow15, N), 'y')
    plt.plot(Reader.sum_range(massflow16, N), 'k')

plt.plot(index1 + cut, arrmf1[index1], 'r+')
plt.plot(index2 + cut, arrmf2[index2], 'r+')
plt.plot(index3 + cut, arrmf3[index3], 'r+')
plt.plot(index4 + cut, arrmf4[index4], 'r+')

if swtch:
    plt.plot(index5 + cut, arrmf5[index5], 'r+')
    plt.plot(index6 + cut, arrmf6[index6], 'r+')

ax2 = plt.subplot(2, 1, 2)
plt.plot(Reader.sum_range(massflow17, N), 'b')
plt.plot(Reader.sum_range(massflow18, N), 'g')
plt.plot(Reader.sum_range(massflow19, N), 'r')
plt.plot(Reader.sum_range(massflow110, N), 'c')

if swtch:
    plt.plot(Reader.sum_range(massflow111, N), 'y')
    plt.plot(Reader.sum_range(massflow112, N), 'k')

plt.plot(index7 + cut, arrmf7[index7], 'r+')
plt.plot(index8 + cut, arrmf8[index8], 'r+')
plt.plot(index9 + cut, arrmf9[index9], 'r+')
plt.plot(index10 + cut, arrmf8[index10], 'r+')

if swtch:
    plt.plot(index11 + cut, arrmf11[index11], 'r+')
    plt.plot(index12 + cut, arrmf12[index12], 'r+')


if swtch:
    trialVec = arange(1, 7)
else:
    trialVec = arange(1, 5)


# 3rd figure - Plotting the average pulse difference
with plt.style.context('seaborn-whitegrid'):
    fig3 = plt.figure()

    plt.subplot(221)
    Uncoated = [0.179150194134, 0.132758413512, 0.133438132927, 0.120245451397, 0.109717925672, 0.0972706894308]
    TwoHundred = [0.640195320928, 0.615280185154, 0.699703628986, 0.678569940659, 0.649157255477, 0.650804396324] # 200 angstrom
    OneThousand = [0.51034669576064506, 0.53275220960590597, 0.54960486142983067, 0.54886651636821415, 0.52639475436452936, 0.53798823223329264]
    TwoThousand = [0.33245350116477707, 0.34580218247722599, 0.29417955765394421, 0.30477963887500531, 0.30716709277010096, 0.30190330598293041]
    plt.title('ABS plus different coating thicknesses')

    plt.plot(trialVec, TwoHundred, 'o', label='20nm')
    fit1 = polyfit(trialVec, TwoHundred, 1)
    line1 = poly1d(polyfit(trialVec, TwoHundred, 1))
    plt.plot(trialVec, line1(trialVec))

    plt.plot(trialVec, OneThousand, 'o', label='100nm')
    fit1 = polyfit(trialVec, OneThousand, 1)
    line1 = poly1d(polyfit(trialVec, OneThousand, 1))
    plt.plot(trialVec, line1(trialVec))

    plt.plot(trialVec, TwoThousand, 'o', label='200nm')
    fit1 = polyfit(trialVec, TwoThousand, 1)
    line1 = poly1d(polyfit(trialVec, TwoThousand, 1))
    plt.plot(trialVec, line1(trialVec))


    plt.plot(trialVec, Uncoated, 'o', label='Uncoated')
    line1 = poly1d(polyfit(trialVec, Uncoated, 1))
    plt.plot(trialVec, line1(trialVec))

    plt.xticks(trialVec)
    plt.ylabel('$\Delta$mg/s')
    plt.legend(loc="upper left", shadow=True, title="Thickness", frameon=False, bbox_to_anchor=[0.735, 0.58])
    plt.xlabel('Trials')
    plt.ylim((0.25, 0.75))
    plt.xlim((0.5, 6.5))

    plt.subplot(223)
    plt.title('ABS plus uncoated')
    plt.plot(trialVec, Uncoated, 'o')
    line2 = poly1d(polyfit(trialVec, Uncoated, 1))
    plt.plot(trialVec, line2(trialVec))
    plt.xticks(trialVec)
    plt.ylabel('$\Delta$mg/s')
    plt.xlabel('Trials')
    plt.ylim((0.09, 0.19))
    plt.xlim((0.5, 6.5))

    coatRange = [0, 200, 1000, 2000]
    #ABSp = [0.0972706894308, 0.639570513566, 0.553243376399, 0.306891038137]
    ABSp = [average(Uncoated), average(TwoHundred), average(OneThousand), average(TwoThousand)]
    PA2200 = [0.817442787282, 0.75219338098, 0.390437786838, 0.176939875426]

    plt.subplot(122)
    plt.title('ABS plus and PA 2200')
    plt.plot(coatRange, ABSp, marker='o', color='r', linestyle='-', label='ABS+')
    plt.plot(coatRange, PA2200, marker='o', color='b', linestyle='-', label='PA 2200')
    line3 = poly1d(polyfit(coatRange[1:4], ABSp[1:4], 1))
    line4 = poly1d(polyfit(coatRange, PA2200, 1))
    # plt.plot(coatRange, line3(coatRange), 'r')
    # plt.plot(coatRange, line4(coatRange), 'b')
    plt.ylabel('$\Delta$mg/s')
    plt.xlabel('Coating thickness')
    plt.legend()
"""

# final plot
plt.show()




