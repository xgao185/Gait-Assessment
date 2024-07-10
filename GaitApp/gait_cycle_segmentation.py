def find_peaks_and_valleys(data_points):
    peak_frames = []
    valley_frames = []

    max_value = max(data_points)
    min_value = min(data_points)

    for i in range(1, len(data_points) - 1):
        if data_points[i] > data_points[i - 1] and data_points[i] > data_points[i + 1]:
            # 当前点比前一个和后一个点都大，是波峰
            if data_points[i] > max_value * 0.35 and (len(peak_frames) == 0 or i - peak_frames[-1] > 20):
                peak_frames.append(i)
        elif data_points[i] < data_points[i - 1] and data_points[i] < data_points[i + 1]:
            # 当前点比前一个和后一个点都小，是波谷
            if data_points[i] < min_value * 0.18 and (len(valley_frames) == 0 or i - valley_frames[-1] > 20):
                valley_frames.append(i)

    # return peak_frames, valley_frames
    return peak_frames
