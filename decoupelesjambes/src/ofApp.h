#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxCvHaarFinder.h"
#include "ofxGui.h"

using namespace ofxCv;
using namespace cv;

class ofApp : public ofBaseApp{

public:
    void setup();
    void update();
    void draw();

    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);

    void findtopdown(); //trouve dans l'image courante la position verticale du dessus de tete et du bas des pieds
    void cutjambes();   //découpe et enregistre l'iamge des jambes en coupant le perso en deux (et oui, ca passe bien).
    void cuthaut();     //découpe et enregistre le tronc
    void neck();        //en cours // pour l'instant coupe la zone du coup pour trouver les contours
    void detectface();  //trouve la tête dans l'image courrante
    void removebkg(Mat src,int outputnumber);  //enlève le fond. la qualité du résultat dépend des params choisis
    void assemble(int jambes, int tronc); //assemble jambe et tronc, numéros des fichiers png


    ofxFloatSlider thresh;      //param pour découpe du coup
    ofxIntSlider scale;         //param pour découpe background
    ofxIntSlider Threshold;     //param pour découpe background
    ofxIntSlider dilateSize;    //param pour découpe background
    ofxIntSlider erosionSize;   //param pour découpe background

    ofxPanel gui;

    ofImage InputImg[5];
    Mat InputcvImg;
    Mat InputcvImgA[4];
    Mat imginf;
    Mat imgsup;
    ofImage Neckcrop;
    Mat NeckcropMat;
    Mat NeckcropctMat;
    std::vector<std::vector<cv::Point> > contours;

    ofDirectory dirjambes;
    ofRectangle border;

    //bkg
    //cv::Mat hsvImg;
    //cv::Mat channel[3];
    //cv::Mat rgbImg;
    //cv::Mat gray;
    //cv::Mat edges;
    //Mat edges;
    //Mat detected_edges;

    //display
    cv::Mat outtest1;
    cv::Mat outtest2;


    int imgsel;

    int Headtop;
    int Footfloor;
    int hauteur;
    int taillegauche;
    int tailledroite;
    int troncgauche;
    int troncdroit;
    int largeurtaille;
    int poscentretaille;
    int largeurtronc;
    int poscentretronc;
    float ratiotroncjambes;
    int halfneckH;




    cv::CascadeClassifier face_cascade;
    std::vector<cv::Rect> faces;
};
