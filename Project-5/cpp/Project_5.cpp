#include <iostream> 
#include <opencv2/opencv.hpp> 
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <thread>

using namespace std;
using namespace cv; 
using namespace Eigen;

int num_images;
int img_width, img_height, pano_img_width;
const float epsilon = 0.6;
const float p = 0.99;
const int rand_num = 6; // Number of correspondences to choose in each trial
const int N = ceil(log(1 - p) / log(1 - pow(1 - epsilon, rand_num)));  // Number of trials
const int delta = 5; // 3*sigma (sigma is set to a small number between 0.5 and 2)
int M; // Threshold of number of inliers

cv::Mat ComputeHomography(const vector<Point2f>& points_1, const vector<Point2f>& points_2, int pairs_num){

    MatrixXd A(pairs_num * 2, 8);
    VectorXd b(pairs_num * 2);

    float x, y, xp, yp;

    for (int i = 0; i < pairs_num; i++){
        x = points_1[i].x;
        y = points_1[i].y;
        xp = points_2[i].x;
        yp = points_2[i].y;

        A.row(i*2) << x, y, 1, 0, 0, 0, -x*xp, -y*xp;
        A.row(i*2+1) << 0, 0, 0, x, y, 1, -x*yp, -y*yp;
        b(i*2, 0) = xp;
        b(i*2+1, 0) = yp;
    }

    Eigen::VectorXd h = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
    cv::Mat H = (Mat_<double>(3,3) << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), 1);

    return H;
}


cv::Mat ComputeHomography_LLS(const vector<Point2f>& points_1, const vector<Point2f>& points_2){

    int pairs_num = points_1.size();
    MatrixXd A(pairs_num * 2, 8);
    VectorXd b(pairs_num * 2);

    float x, y, xp, yp;
    RowVectorXd newRow(8);

    for (int i = 0; i < pairs_num; i++){
        x = points_1[i].x;
        y = points_1[i].y;
        xp = points_2[i].x;
        yp = points_2[i].y;

        A.row(i*2) << x, y, 1, 0, 0, 0, -x*xp, -y*xp;
        A.row(i*2+1) << 0, 0, 0, x, y, 1, -x*yp, -y*yp;
        b(i*2, 0) = xp;
        b(i*2+1, 0) = yp;
    }

    VectorXd h = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    cv::Mat H_LLS = (Mat_<double>(3,3) << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), 1);

    return H_LLS;
}


cv::Mat getRefinedHomographyFromImagePair(const Mat img1, const Mat img2){

    // Convert images to grayscale
    Mat img1_gray, img2_gray;
    cvtColor(img1, img1_gray, COLOR_BGR2GRAY);
    cvtColor(img2, img2_gray, COLOR_BGR2GRAY);

    // SIFT feature detector and descriptor
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    detector -> detectAndCompute(img1_gray, noArray(), keypoints_1, descriptors_1);
    detector -> detectAndCompute(img2_gray, noArray(), keypoints_2, descriptors_2);

    // BFMatcher with default params
    BFMatcher matcher(NORM_L1, true);
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    int n_matches = matches.size();

    // Sort matches by distance
    sort(matches.begin(), matches.end(), [](const DMatch& a, const DMatch& b) {
        return a.distance < b.distance;
    });

    // Store coordinates
    vector<Point2f> matchpoints_1, matchpoints_2;
    for (const auto& match : matches) {
        matchpoints_1.push_back(keypoints_1[match.queryIdx].pt);
        matchpoints_2.push_back(keypoints_2[match.trainIdx].pt);
    }

    // Reject outliers with RANSAC
    M = static_cast<int>(ceil((1 - epsilon) * n_matches)); // Threshold of number of inliers

    int inliers_num = 0;
    vector<Point2f> matchpoints_1_RANSAC, matchpoints_2_RANSAC;
    random_device rd;
    mt19937 g(rd());
    for (int i = 0; i < N; ++i) { 
        vector<Point2f> matchpoints_1_random, matchpoints_2_random;

        // Random sampling
        vector<int> indices(n_matches);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), g);
        for (int j = 0; j < rand_num; ++j) {
            matchpoints_1_random.push_back({matchpoints_1[indices[j]]});
            matchpoints_2_random.push_back({matchpoints_2[indices[j]]});
        }

        // Compute initial homography
        cv::Mat H_initial = ComputeHomography(matchpoints_1_random, matchpoints_2_random, rand_num);

        // Find inliers
        vector<int> current_inliers;
        std::vector<Point2f> projected_points(n_matches);
        cv::perspectiveTransform(matchpoints_1, projected_points, H_initial);
        
        vector<Point2f> matchpoints_1_inliers, matchpoints_2_inliers;
        for (int i = 0; i < n_matches; i++){
            double error = norm(matchpoints_2[i] - projected_points[i]);
            if (error <= pow(delta, 2)) {
                matchpoints_1_inliers.push_back(matchpoints_1[i]);
                matchpoints_2_inliers.push_back(matchpoints_2[i]);
            }
        }

        if (matchpoints_1_inliers.size() > inliers_num) {
            inliers_num = matchpoints_1_inliers.size();
            matchpoints_1_RANSAC = matchpoints_1_inliers;
            matchpoints_2_RANSAC = matchpoints_2_inliers;
        }

        if (current_inliers.size() > M) {
            break;
        }
    }

    Mat H_LLS = ComputeHomography_LLS(matchpoints_1_RANSAC, matchpoints_2_RANSAC);

    return H_LLS;
}


void projectOneImage(cv::Mat pano_img, const cv::Mat src_img, const cv::Mat H, const bool debug){

    // Project img to the panorama image
    std::vector<Point2f> obj_pixel(1);
    std::vector<Point2f> scene_pixel(1);
    Point2f offset = {static_cast<float>(img_width*((num_images-1)/2)), 0};

    for (int x = 0; x < img_width; x++){ 
        for (int y = 0; y < img_height; y++){ 
            
            obj_pixel[0] = cv::Point(x, y);
            
            perspectiveTransform(obj_pixel, scene_pixel, H);
            scene_pixel[0] = scene_pixel[0] + offset;
            int scene_pixel_x = round(scene_pixel[0].x);
            int scene_pixel_y = round(scene_pixel[0].y);

            if ((scene_pixel_x < pano_img_width) && (scene_pixel_y < img_height) && (pano_img.at<cv::Vec3b>(scene_pixel_y, scene_pixel_x) == cv::Vec3b(0, 0, 0))){
                if ((0 <= scene_pixel_x) && (scene_pixel_x < pano_img_width) && (0 <= scene_pixel_y) && (scene_pixel_y < img_height)) {
                    if (debug) cout << "obj_pixel = " << obj_pixel << ", scene_pixel = [" << scene_pixel_x << ", " << scene_pixel_y << "]\n";
                    pano_img.at<cv::Vec3b>(scene_pixel_y, scene_pixel_x) = src_img.at<cv::Vec3b>(y, x);
                }
                // fill the holes
                if ((0 < scene_pixel_x) && (scene_pixel_x < pano_img_width-1) && (0 < scene_pixel_y) && (scene_pixel_y < img_height-1)){
                    pano_img.at<cv::Vec3b>(scene_pixel_y+1, scene_pixel_x) = src_img.at<cv::Vec3b>(y, x);
                    pano_img.at<cv::Vec3b>(scene_pixel_y-1, scene_pixel_x) = src_img.at<cv::Vec3b>(y, x);
                    pano_img.at<cv::Vec3b>(scene_pixel_y, scene_pixel_x+1) = src_img.at<cv::Vec3b>(y, x);
                    pano_img.at<cv::Vec3b>(scene_pixel_y, scene_pixel_x-1) = src_img.at<cv::Vec3b>(y, x);
                }
            }
        }
    }
    cout << "\nProject image done.\n";
}


int main(){

    cout << "\nStarting Project 5 ... \n";
    num_images = 5;

    // Read in images
    Mat img0 = imread("0.jpg");
    Mat img1 = imread("1.jpg");
    Mat img2 = imread("2.jpg");
    Mat img3 = imread("3.jpg");
    Mat img4 = imread("4.jpg");

    if (img1.empty()) {
        std::cerr << "Image 1 not found" << std::endl;
        return -1;
    }

    img_width = img1.size().width;   // 747
    img_height = img1.size().height; // 1328
    pano_img_width = img_width * num_images; // 2241

    // Declare homography matrices
    cv::Mat H_01, H_12, H_23, H_34;

    // Launch threads for each homography computation
    std::thread t1([&]{ H_01 = getRefinedHomographyFromImagePair(img0, img1); });
    std::thread t2([&]{ H_12 = getRefinedHomographyFromImagePair(img1, img2); });
    std::thread t3([&]{ H_23 = getRefinedHomographyFromImagePair(img2, img3); });
    std::thread t4([&]{ H_34 = getRefinedHomographyFromImagePair(img3, img4); });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // Create a new image with the total width of the two images
    Mat pano_img(img_height, pano_img_width, CV_8UC3, cv::Scalar(0, 0, 0));

    // Place middle image
    img2.copyTo(pano_img(Rect(img_width*int((num_images)/2), 0, img_width, img_height)));
    cout << "\nMiddle img copy to middle\n";

    // Project images
    projectOneImage(pano_img, img1, H_12, false);
    projectOneImage(pano_img, img0, H_01 * H_12, false);
    projectOneImage(pano_img, img3, H_23.inv(), false);
    projectOneImage(pano_img, img4, (H_23 * H_34).inv(), false);

    cv::imwrite("Pano_image.png", pano_img);
    
    return 0;
}
