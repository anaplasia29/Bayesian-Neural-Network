import numpy as np
import os

def subthreshold_swing(logId):
    """
    logId 라는 array를 받아와서 sample 행렬을 만듦.
    logId를 역수를 취한 후, -10을 곱함. -> -10/logId 
    subth = 0.001으로 시작하여 sample에 대해서
    전체 반복을 하여 subth보다 sample[i][j+1] - sample[i][j]가 더 클 경우에
    subth를 바꾼다. 그렇게 각 row에 대하여 subth value를 구한 후
    subth_value는 0.05 / subth로 하여, 전체 반복을 한다.
    반복 완료 후 subth_array를 return. 
    """
    sample = np.array(logId)
    sample = sample.astype(np.float)
    rowsize = sample.shape[0]
    columsize = sample.shape[1]

    # for i in range(0, rowsize) :
    #     for j in range(0,columsize) :
    #         sample[i][j] = -10/sample[i][j]
    sample = np.reciprocal(sample)
    sample = sample * (-10)
    subth_array = []

    for i in range(0, rowsize) :
        subth = 0.001
        for j in range(0, columsize - 1) :
            if subth < (sample[i][j+1] - sample[i][j]) :
                subth = sample[i][j+1] - sample[i][j]
        subth_value = 0.05 / subth
        subth_array.append(subth_value)
        np_subth_array = np.array(subth_array)
    return np_subth_array

def threshold_voltage(logId, print_error = True):
    """
    기울기가 커야 전압을 조금만 조절해도 전류 변화가 크고, 
    누설 전류가 작음. 
    """
    sample = np.array(logId)
    sample = sample.astype(np.float)
    rowsize = sample.shape[0]
    columsize = sample.shape[1]

    for i in range(0, rowsize) :
        for j in range(0,columsize) :
            if sample[i,j] != 0 :
                sample[i,j] = -10/sample[i,j]
            else :
                sample[i, j] = -10 / (sample[i, j]+1e-10)

    real_value_sample = np.zeros((rowsize, columsize) , dtype= float)
    real_value_sample = np.exp(np.log(10)*sample)
    sample_gradient = np.zeros((rowsize, columsize -1), dtype=float)
    for j in range(1,columsize) :
        sample_gradient[:,j-1] = 20 * (real_value_sample[:,j]- real_value_sample[:,j-1])

    gcp = np.zeros((rowsize, columsize-2), dtype=int)
    # gcp is gradient_compare_parameter
    threshold_voltage_array = np.zeros((rowsize), dtype=float)

    for i in range(0, rowsize) :
        for j in range(0, columsize) :
            # real_value_sample[i,j] = np.exp(np.log(10)*sample[i,j])
            # if j > 0:
            #     sample_gradient[i,j-1] = 20*(real_value_sample[i,j]-real_value_sample[i,j-1])
            if j > 1:
                if sample_gradient[i,j-1]/sample_gradient[i,j-2] > 0.75 and sample_gradient[i,j-1]/sample_gradient[i,j-2] < 1.25 :
                    gcp[i][j-2] = 1
                    # print("gcp 의 " + str(i) + "행 " + str(j-2) + "열의 값은 1입니다"  )
        for k in range(0, columsize - 5) :
            if sample[i,k] > -7.5 and gcp[i][k] == 1 and  gcp[i][k+1] == 1 and gcp[i][k+2] == 1 and k < 35:
                threshold_voltage_array[i] = (0.05 * k - 1) - (real_value_sample[i][k]/sample_gradient[i][k])
                # print(str(i+1)+ "번째 sample의 접점은 " + str(50*k -1000) + "mV 와 그때의 voltage 값은 " + str(real_value_sample[i][k]) )
                # print(real_value_sample[i][k])
                # print(sample_gradient[i][k])
        if threshold_voltage_array[i] < 0 :
            threshold_voltage_array[i] = 0
        for l in range(0, columsize -1) :
            if sample_gradient[i][l] == 0 and print_error is True:
                print(str(i) + " 번째 샘플에 문제가 있습니다. 기울기가 0으로 측정됨")

    return threshold_voltage_array

def mobility_degradation(sample_mueff, threshold_voltage , vgt_float=1.0):
    """
    vgt_float : 내가 얼마나 띄울 것인지 입력, default : 1.0
    mobility가 감소하는 것이기 때문에 낮으면 좋을 것으로 예상된다.
    """
    mueff = np.array(sample_mueff)
    mueff = mueff.astype(np.float)
    vth = np.array(threshold_voltage)
    vth = vth.astype(np.float)
    vth = np.reshape(vth, -1)
    rowsize = mueff.shape[0]
    columnsize = mueff.shape[1]
    vgt_effective_mueff_array = []
    vgt = float(vgt_float)
    for i in range (0, rowsize) :
        max_mueff = 0
        effective_mueff = 1
        voltage_max_mueff = 0
        check = 0
        for j in range(0, columnsize) :
            if mueff[i,j] > max_mueff :
                max_mueff = mueff[i,j]
                voltage_max_mueff = j
        for k in range(0, columnsize) :
            if k > voltage_max_mueff and (0.05 * k) - vth[i] - 1 > vgt - 0.02501 and (0.05 * k) -vth[i] - 1 < vgt + 0.02501  :
                effective_mueff = mueff[i,k] / max_mueff
                check = 1
            if k == columnsize - 1 and check == 0 :
                effective_mueff = mueff[i,k] / max_mueff
        vgt_effective_mueff_array.append(effective_mueff)
    mueff_array = np.array(vgt_effective_mueff_array)
    return mueff_array

"""
여기서 부터는 나는 쓸모없는 함수들
"""
def R_square(real_y , predict_y):
    """ 여기서 쓰려고 만듦
    Args:
        real_y (list): [description]
        predict_y (list): [description]

    Returns:
        [type]: [description]
    """
    real = np.array(real_y).reshape(-1)
    predict = np.array(predict_y).reshape(-1)
    if predict.shape[0] == real.shape[0] :
        mean  = np.mean(real)
        std = np.std(real)
        samplesize = real.shape[0]
        unexplained_error = 0
        total_error = std*std
        for i in range (0, samplesize) :
            unexplained_error = unexplained_error +(real[i] - predict[i]) * (real[i] - predict[i])
        unexplained_error = unexplained_error/samplesize
            # unexplained_error = tf.reduce_mean((original_Y - original_Y_predict) * (original_Y - original_Y_predict))
        if total_error == 0 :
            return 1
        R_square_value = 1 - (unexplained_error / total_error)
        return R_square_value

def RMSE(real_y, predict_y):
    real = np.array(real_y).reshape(-1)
    predict = np.array(predict_y).reshape(-1)
    if predict.shape[0] == real.shape[0] :
        rmse = (real - predict) * (real - predict)
        rmse = np.mean(rmse)
        rmse = np.sqrt(rmse)
    return rmse

def array_log(real_value, base , array_rank) :
    base_num = int(base)
    if array_rank == 2 :
        real = np.array(real_value)
        rowsize = real.shape[0]
        columnsize = real.shape[1]
        return_array = np.zeros((rowsize, columnsize), dtype=float)
        for i in range(0, rowsize) :
            for j in range(0, columnsize) :
                return_array[i][j] = np.log(real[i][j], base_num)
        return  return_array

    if array_rank == 1 :
        real = np.array(real_value)
        rowsize = real.shape[0]
        return_array = np.zeros(rowsize, dtype=float)
        for i in range(0, rowsize) :
            return_array[i] = np.log(real[i], base_num)
        return  return_array


def array_exp(log_value, base, array_rank) :
    base_num = int(base)
    if array_rank == 2 :
        log = np.array(log_value)
        rowsize= log.shape[0]
        columnsize = log.shape[1]
        return_array = np.zeros((rowsize, columnsize), dtype=float)
        for i in range(0, rowsize):
            for j in range(0, columnsize):
                return_array[i][j] = np.exp(log[i][j] * np.log(base_num))
        return return_array

    if array_rank == 1 :
        log = np.array(log_value)
        rowsize= log.shape[0]
        print(rowsize)
        return_array = np.zeros(rowsize, dtype=float)
        for i in range(0, rowsize) :
            return_array[i] = np.exp(log[i] * np.log(base_num))
        return return_array


def txt_to_value(directory, sample_number, type) :
    file_dir = directory
    sample_name_list = os.listdir(file_dir)
    width = sample_name_list[0].split('W')[1].split('T')[0]
    tsi = sample_name_list[0].split('T')[1].split('V')[0]
    vbg = sample_name_list[0].split('V')[1].split('.t')[0]

    ava = []

    for i in range(0, sample_number):
        width = sample_name_list[i].split('W')[1].split('T')[0]
        tsi = sample_name_list[i].split('T')[1].split('V')[0]
        vbg = sample_name_list[i].split('V')[1].split('.t')[0]
        file_name = sample_name_list[i]
        data_read = open(file_dir + "/" + file_name, 'r')
        header = -1
        data_set = []
        data_set.append(width)
        data_set.append(tsi)
        data_set.append(vbg)
        while True:
            line = data_read.readline()
            if not line: break
            if line == "{\n":
                header = 1
            elif line == "}\n":
                header = 0
            elif header == 0:
                line.strip()
                split_line = line.split()
                if type == 0 :
                    log_value = split_line[1].split('e-')
                    log_value_10 = -10 / (np.log10(float(log_value[0])) + (-1) * float(log_value[1]))
                    data_set.append(log_value_10)
                # code for V/Id , V/Qn
                else :
                    log_value = split_line[1]
                    log_value_10 = log_value
                    data_set.append(log_value_10)
                # code for mueff


                # log_value_10 = -10/(np.log10(float(log_value[0])) + (-1) * float(log_value[1]))
                # code for V/Id , V/Qn

                # log_value_10 = log_value
                # code for mueff
        data_set = np.array(data_set)
        ava.append(data_set[0:53])
    retrun_array = np.array(ava)
    retrun_array = retrun_array.astype(np.float)
    return retrun_array

def linear_smoothing(log_Id) :
    log_Id = np.array(log_Id)
    linear_region = 10
    Vg = range(-20, 30)
    Vg = np.array(Vg)
    Vg = Vg / 20
    real_Id = -10 * np.reciprocal(log_Id)
    real_Id = np.reshape(real_Id, -1)
    smooth_log_Id = log_Id.copy()
    smooth_Id = np.exp(real_Id*np.log(10))
    linear_rsq = -100
    i_start = 0
    smooth_coef = np.zeros(2,dtype=float)
    for i in range(30) :
        if real_Id[i] > -7.5 :
            linear_coef = np.polyfit(Vg[i:i+linear_region], smooth_Id[i:i+linear_region],1)
            linear_smooth = linear_coef[0]* Vg[i:i+linear_region] + linear_coef[1]
            rsq = R_square(smooth_Id[i:i+linear_region],linear_smooth)
            if rsq > linear_rsq :
                i_start = i
                smooth_coef[0] = linear_coef[0]
                smooth_coef[1]= linear_coef[1]

    smooth_Id[i_start: i_start+linear_region] = smooth_coef[0] * Vg[i_start:i_start+linear_region] + smooth_coef[1]
    smooth_Id = np.reciprocal(np.log10(smooth_Id)/(-10))
    return smooth_Id