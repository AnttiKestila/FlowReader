from numpy import interp, linspace, pi, sqrt, array
import operator
import peakutils

class Reader:
    """This a collection of the tools needed for printing out sensible output for the flow setup"""

    def __init__(self, actualdata, calibrationdata):
        self.actualdata = actualdata
        self.calibrationdata = calibrationdata

    # Quick interpolation of mass flow
    def massinterp(self, x_massflow, y_data_massflow, Tc):

        # Interpolating mass flow values and argon density

        rho = 39.9 / (0.08206 * (Tc + 273.15))

        # Calibration based on initial values of the mass flow sensors

        f1 = self.actualdata[1, 2]
        f2 = self.actualdata[1, 3]
        difference = abs(f1 - f2)

        # conversion into

        if difference >= 0:
            d1 = -difference/2
            d2 = difference/2
        else:
            d1 = difference/2
            d2 = -difference/2

        m1c = [x * rho for x in (interp(self.actualdata[:, 2] + d1, x_massflow, y_data_massflow) * 1000 / 60)]  # interp gives l/min so division by 60 to get down to per sec]ond
        m2c = [x * rho for x in (interp(self.actualdata[:, 3] + d2, x_massflow, y_data_massflow) * 1000 / 60)]

        return m1c, m2c

    @staticmethod
    def sum_range(sumdata, N):
        """ sums values into one every N length window"""
        firstlen = len(sumdata)
        compresseddata = [sum(sumdata[i:min(i + N, firstlen)]) for i in range(firstlen)]

        return compresseddata

    @staticmethod
    def peaking(mf, thresh, min_dist):
        arrmf = array(mf)
        index = peakutils.indexes(arrmf, thresh, min_dist)
        return arrmf, index

    @staticmethod
    def interpressure(actual, ref, uplim, downlim, channel):  # Interpolating the pressure into bars
        backbone = linspace(uplim, downlim, 1 / 0.005)  # backbone for the pressure range
        caliban = ((ref[:, channel] - ref[-1, channel]) / ((ref[0, channel] - ref[-1, channel]) * 0.005) + 1) * (backbone[-2] - backbone[-1])
        sorted_p = sorted(set(zip(ref[:, channel], caliban)), key=operator.itemgetter(1))
        divided_p, divided_caliban = zip(*sorted_p)
        calibrated = interp(actual[:, channel], divided_p, divided_caliban)

        return calibrated

    # Artificial mass flow according to NASA's critical flow equation for venturi
    @staticmethod
    def pressure2mf(diameter, pt, Tin):
        gamma = 1.6696  # for Argon
        R = 208  # for Argon
        tt = 273.15 + Tin
        outmf = 1000 * 1000 * ((pi * (diameter / 2)**2) * (100000 * pt) / sqrt(tt)) * sqrt(gamma / R) * (1 + (gamma - 1) / 2)**(-1 * (gamma + 1) / (2 * (gamma - 1)))

        return outmf

    def plotter(self, ip, thrtd, x_mflow, y_data_mflow, Tc, calibration):
        p1c = self.interpressure(self.actualdata, self.calibrationdata, ip, 0.0000065, 0)
        p2c = self.interpressure(self.actualdata, self.calibrationdata, ip, 0.0000065, 1)

        massflow1, massflow2 = self.massinterp(x_mflow, y_data_mflow, Tc)
        massflow1 = [x + calibration for x in massflow1]
        m1cc = self.pressure2mf(thrtd, p1c, Tc)

        return p1c, p2c, massflow1, massflow2, m1cc

