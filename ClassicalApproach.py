from __future__ import division
import sys
import math
import numpy as np
import cv2
from pylsd.lsd import lsd
from glob import glob
import pandas as pd
from openpyxl import load_workbook
## CLASSICAL METHOD


def thetam(pt1, pt2):
    t1, t2 = pt1[5], pt2[5]  # Lines corressponding to leftmost and rightmost.

    a1 = pt1[0]
    b1 = pt1[1]
    a2 = pt1[2]
    b2 = pt1[3]

    x1 = pt2[0]
    y1 = pt2[1]
    x2 = pt2[2]
    y2 = pt2[3]

    theta1 = abs(np.arctan((b1 - b2) / (a1 - a2)))
    theta2 = abs(np.arctan((y1 - y2) / (x1 - x2)))

    theta1 = math.pi / 2 - theta1
    theta2 = -1 * (math.pi / 2 - theta2)

    m1 = np.tan(theta1)
    m2 = np.tan(theta2)

    m = m1 + m2

    atan = m / 2

    tm = math.atan(atan)
    return tm


def angle(pt1, pt2):
    t1, t2 = pt1[5], pt2[5]
    # if t1<0:
    #     t1 = t1 + math.pi
    # if t2<0:
    #     t2 = t2 + math.pi
    if t1 < 0:
        t1 = math.pi + t1
    if t2 < 0:
        t2 = math.pi + t2

    # print "The theta values are {0}, {1}".format(t1*180/math.pi,t2*180/math.pi)

    tb = abs(t1 - t2)
    ret = (min(t1, t2) + (tb / 2))

    # print 'The return value is {0}'.format(ret)

    return ret


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def setvp(event, x, y, flags, param):
    global v_x, v_y, xfin, yfin, showagain, k

    showagain = 1

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left is {},{}".format(x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        print(x,y)
        v_x = x
        v_y = y
        # cv2.destroyAllWindows()
        # k = 27

    if event == cv2.EVENT_RBUTTONDOWN:
        print("Right is {},{}".format(x, y))

    elif event == cv2.EVENT_RBUTTONUP:
        xfin = x
        yfin = y


## This function finds the theta for each line and extend these lines
## to intersect with the images borders and then make sure that they are
## intersected to be fed into the intesection function later.
def process(lines, img, v_x, v_y, vp_number):
    ii = 1
    sel_lines = []
    thetas = []
    mags = []
    j = 0
    w = 2
    xrange = range
    for i in xrange(lines.shape[0]):

        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))

        x1 = int(lines[i, 0])
        x2 = int(lines[i, 2])
        y1 = int(lines[i, 1])
        y2 = int(lines[i, 3])
        w = int(lines[i, 4])

        # for pt1,pt2 in lines:
        #     if y1 < y_mean or y2 < y_mean:
        #         lines.remove(pt1)

        if (x1 - x2) != 0:  # Make sure lines are not collinear
            theta = np.arctan2((y2 - y1), (x2 - x1))

            m2 = np.tan(theta)
            l_mag = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

            # Extend the lines to the entire image and compute the intersetion point
            c2 = y1 - m2 * x1
            x3 = int(img.shape[1] / 1.8 + x1)  # 1000 was chosen arbitrarily (any number higher than half the image width)
            y3 = int(m2 * x3 + c2)
            x4 = int(x1 - img.shape[1] / 1.8)  # 1000 was chosen arbitrarily
            y4 = int(m2 * x4 + c2)

            # if y4<v_y:
            #     y4 = int(v_y)

            lines1 = lines[i]
            # if vp_number == "VP1":
            #     range_condition = l_mag > 1 and 0.1 < abs(theta) < 1.4
            # elif vp_number == "VP2":
            #     range_condition = l_mag > 20 and 0.2 < abs(theta) < 1.1
            # elif vp_number == "VP3":
            #     range_condition = l_mag > 30 and 0.2 < abs(theta) < 1.1
            # elif vp_number == "VP4":
            #     range_condition = l_mag > 40 and 0.3 < abs(theta) < 1.3
            # elif vp_number == "VP5":
            #     range_condition = l_mag > 25 and 0.3 < abs(theta) < 1.3
            # elif vp_number == "VP6":
            #     range_condition = l_mag > 20 and 0.3 < abs(theta) < 1.4
            # elif vp_number == "VP7":
            #     range_condition = l_mag > 20 and 0.3 < abs(theta) < 1.3
            # elif vp_number == "VP8":
            #     range_condition = l_mag > 20 and 0.2 < abs(theta) < 1.0
            # elif vp_number == "VP9":
            #     range_condition = l_mag > 15 and 0.3 < abs(theta) < 1.1
            # elif vp_number == "VP10":
            #     range_condition = l_mag > 10 and 0.3 < abs(theta) < 1.1
            # elif vp_number == "VP11":
            #     range_condition = l_mag > 10 and 0.2 < abs(theta) < 1.4
            # elif vp_number == "VP12":
            #     range_condition = l_mag > 1 and 0.3 < abs(theta) < 1.3
            # elif vp_number == "VP13":
            #     range_condition = l_mag > 5 and 0.1 < abs(theta) < 0.9
            # elif vp_number == "VP14":
            #       range_condition = l_mag > 10 and 0.1 < abs(theta) < 0.8
            # elif vp_number == "VP15":  # Detect VP Manually
            # #       range_condition = l_mag > 20 and 0.1 < abs(theta) < 1.4
            # #       range_condition = l_mag > 1 and 0.1 < abs(theta) < 1.4
            #         range_condition = True
            # else:
            range_condition = True

            if range_condition:
                lines1 = np.append(lines1, theta)
                lines1 = np.append(lines1, l_mag)
                thetas = np.append(thetas, theta)
                mags = np.append(mags, l_mag)

                sel_lines.append(lines1)

                if y3 > v_y or y4 > v_y:
                    if y3 >= y4:
                        y4 = v_y
                        x4 = int((y4 - c2) / m2)
                        # print 'y3 > y4: {0}, {1}, {2}, {3}'.format(x3,y3,x4,y4)
                        pt11 = (x4, y4)
                        pt22 = (x3, y3)
                    else:
                        y3 = v_y
                        x3 = int((y3 - c2) / m2)
                        # print 'y4 > y3: {0}, {1}, {2}, {3}'.format(x3,y3,x4,y4)
                        pt11 = (x3, y3)
                        pt22 = (x4, y4)

                sel_lines[j][0] = int(x3)
                sel_lines[j][1] = int(y3)
                sel_lines[j][2] = int(x4)
                sel_lines[j][3] = int(y4)

                j = j + 1

    return sel_lines, thetas, mags, w


def classical(img, image_name, vp_number):
    global v_x, v_y, xfin, yfin, showagain, k
    img = cv2.resize(img, (224, 224))  # For Resnet18
    img = cv2.flip(img, 1)

    v_x = int(img.shape[1] / 2)
    v_y = int(img.shape[0] / 2)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", setvp)

    while True:
        cv2.imshow("image", img)

        k = cv2.waitKey(0) & 0xFF

        if k == 27:
            cv2.destroyAllWindows()
            break

    sss = "H:/Desktop/CARA/Dataset/Classic - Campus/{0}".format(image_name)

    showagain = 1

    if showagain == 1:

        showagain = 0

        yfin = img.shape[0]
        xfin = img.shape[1]/2
        cv2.line(img, (int(v_x), int(v_y)), (int(xfin), int(yfin)), (0, 255, 0), 4)

        cv2.circle(img, (int(v_x), int(v_y)), 10, (0, 255, 255), 3)

        #Drawing the desired line in black
        cv2.line(img, (int(img.shape[1] / 2), 0), (int(img.shape[1] / 2), img.shape[0]), (0, 0, 0), 3)

        den = (xfin-v_x)

        if den == 0:
            den = 1

        theta = np.arctan((yfin-v_y)/den)

        if theta < 0:
            theta = -math.pi/2 - theta
        if theta > 0:
            theta = math.pi/2 - theta

        theta_m = theta

        img_original = cv2.flip(img, 1)
        cv2.imwrite(sss, img_original)

    print("TM is {}".format(theta_m*180/math.pi))

    v_x = v_x - (img.shape[1] / 2)   # CHANGING TO CARTESIAN COORDINATES AS GIVEN IN THE PAPER
    v_y = -(v_y - (img.shape[0] / 2))

    v_x = v_x * (1 / 5675)  # Converting pixels to meters. Scale is 1m = 5675 pixels
    v_y = v_y * (1 / 5675)

    h = 1.6  # 0.47 for umich
    l = 0

    w = 0

    s = np.sin(theta_m)
    c = np.cos(theta_m)

    pm = v_x * c + v_y * s

    lambda_m = np.cos(theta_m) / h

    print("V_X is {}".format(v_x))

    error = [[v_x], [theta_m]]

    error = np.matrix(error)

    le = 100 * error

    Jw = [[1 + np.square(v_x)], [((-1) * lambda_m * l * c) + (lambda_m * w * pm) + (pm * s)]]

    Jw = np.matrix(Jw)

    Jv = [[0], [(-1) * (lambda_m * pm)]]

    Jv = np.matrix(Jv)

    vconst = 0.2

    pinv = (-1) * (np.linalg.pinv(Jw))

    fmat = le + Jv * vconst

    w = pinv * fmat

    print('w is {0} \n'.format(w))

    w = float(w)

    return w, img, v_x, v_y


def main():
    path = "H:\\Desktop\\CARA\\Dataset\\Campus\\4\\*.png"
    images_paths = glob(path)

    # Read csv into dataframe
    csv_name = 'H:/Desktop/CARA/Dataset/Vanishing points - Campus.xlsx'
    df = pd.read_excel(csv_name, sheet_name="VP All")
    wdf = pd.read_excel(csv_name, sheet_name="w All")
    writer = pd.ExcelWriter(csv_name, engine='openpyxl', mode="a")
    writer.book = load_workbook(csv_name)
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    vp_number = "15"
    for ii, image_path in enumerate(images_paths):

        print(ii)
        img = cv2.imread(image_path)
        image_name = image_path.split("\\")[-1]
        # if image_name not in ("5498.png", "5499.png", "5500.png", "5501.png"):
        #     continue
        w, img1, v_x, v_y = classical(img, image_name, "VP" + vp_number)
        df.loc[df["Image"] == image_name, "VP"] = np.round(v_x, 6)
        df.loc[df["Image"] == image_name, "VY"] = np.round(v_y, 6)
        wdf.loc[wdf["Image"] == image_name, "w"] = np.round(w, 6)
        print("w classical is {}".format(w))

    df.to_excel(writer, sheet_name="VP All", startrow=1, header=False, index=False)
    wdf.to_excel(writer, sheet_name="w All", startrow=1, header=False, index=False)
    writer.save()


def main2():
    readPath = "H:\\Desktop\\CARA\\Dataset - Full\\"

    # Read csv into dataframe
    csv_name = 'H:/Desktop/CARA/Intelligent-Wheelchair-Platform-master/IROS scripts/Vanishing points (x).xlsx'
    vpdf = pd.read_excel(csv_name, sheet_name="GT VP")
    images_names = ["2749.png"]#"vpdf["Image"]
    temp = []
    for ii, image_name in enumerate(images_names):
        print(ii)
        print(image_name)
        try:
            image_path = readPath + image_name
            img = cv2.imread(image_path)
            if img is None:
                continue
            vp_number = vpdf.loc[vpdf["Image"] == image_name, "VP number"].values

            w, img1, v_x = classical(img, image_name, vp_number)
            #print("w classical is {}".format(w))

        except:
            temp.append(image_name)

    print(temp)


if __name__ == '__main__':
    ino = main()
