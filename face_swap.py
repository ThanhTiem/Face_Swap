import cv2
import numpy as np
import dlib

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def swap_face(img1, img2):
    #convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #create a marsk (balck image with same size as original image)
    mask = np.zeros_like(gray1)

    #detect face and facial landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model\shape_predictor_68_face_landmarks.dat')
    height, width, channels  = img2.shape
    img2_new_face  = np.zeros((height, width, channels), np.uint8)

    #face 1
    faces1 = detector(gray1)
    for face in faces1:
        landmarks = predictor(gray1, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            #cv2.circle(img1, (x,y),3,(0,0,255), -1)
        #cv2.imshow("face swapped", img1)
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convexhull, 255)
        face_image_1 = cv2.bitwise_and(img1, img1, mask=mask)
        #cv2.imshow("face swapped", img1)

        #delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            #cv2.line(img1, pt1, pt2, (0, 0, 255), 1)
            #cv2.line(img1, pt2, pt3, (0, 0, 255), 1)
            #cv2.line(img1, pt1, pt3, (0, 0, 255), 1)

            index_pt1 = np.where((points==pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)
        #cv2.imshow("face swapped", img1)


    #face 2
    faces2 = detector(gray2)
    for face in faces2:
        landmarks = predictor(gray2, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))

        points2 = np.array(landmarks_points2, np.int32)
        convexhull2  =cv2.convexHull(points2
    )

    lines_space_mask = np.zeros_like(gray1)
    lines_space_new_face = np.zeros_like(img2)

    #triangulation of both faces
    for triangle_index in indexes_triangles:


        #triangulation of first face
        f1_pt1 = landmarks_points[triangle_index[0]]
        f1_pt2 = landmarks_points[triangle_index[1]]
        f1_pt3 = landmarks_points[triangle_index[2]]
        triangle1  = np.array([f1_pt1, f1_pt2, f1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img1[y: y + h, x: x + w]
        cropped_f1_mask = np.zeros((h, w), np.uint8)


        points = np.array([[f1_pt1[0] - x, f1_pt1[1] - y],
                        [f1_pt2[0] - x, f1_pt2[1] - y],
                        [f1_pt3[0] - x, f1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_f1_mask, points, 255)

        #lines spaces
        cv2.line(lines_space_mask, f1_pt1, f1_pt2, 255)
        cv2.line(lines_space_mask, f1_pt2, f1_pt3, 255)
        cv2.line(lines_space_mask, f1_pt1, f1_pt3, 255)
        lines_space = cv2.bitwise_and(img1, img1, mask=lines_space_mask)

        #triangulation of second face
        f2_pt1 = landmarks_points2[triangle_index[0]]
        f2_pt2 = landmarks_points2[triangle_index[1]]
        f2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([f2_pt1, f2_pt2, f2_pt3], np.int32)


        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_f2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[f2_pt1[0] - x, f2_pt1[1] - y],
                            [f2_pt2[0] - x, f2_pt2[1] - y],
                            [f2_pt3[0] - x, f2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_f2_mask, points2, 255)

        #warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_f2_mask)

        #reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area


    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(gray2)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)


    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    return seamlessclone

