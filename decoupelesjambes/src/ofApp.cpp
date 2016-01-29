#include "ofApp.h"
using namespace ofxCv;
using namespace cv;
//--------------------------------------------------------------
void ofApp::setup(){


    //setup GUI
    gui.setup();
    gui.add(thresh.setup("Thresholdneck",100,0,255));
    gui.add(scale.setup("scale (pour bkg)",1,0,100));
    gui.add(Threshold.setup("Threshold bkg",25,0,255));
    gui.add(dilateSize.setup("dilatesize (pour bkg)",2,0,5));
    gui.add(erosionSize.setup("erosesize (pour bkg)",5,0,10));

    //charge première image
    string pathfile;
    for (int i=0;i<3;i++)
    {   pathfile=std::to_string(i)+".png";
        InputImg[i].load(pathfile);}

    //image sélectionnée, courrante (par numéro de fichier png), change sur clic
    imgsel=0;

    //charger fichier xml pour detection des visages
    if (face_cascade.load("/home/rup/Bureau/of_v0.9.0_linux64_release/apps/myApps/decoupelesjambes/bin/data/haarcascade_frontalface_default.xml"))
    {cout<<"face xml loaded"<<endl;}
    else {cout<<"face xml not loaded"<<endl;}


    //cv::Rect neckrect(0,0,500,100);

    //initialiser qques vars
    bool found=FALSE;
    Headtop=0;
    hauteur=0;
    Footfloor=0;
    InputcvImg=toCv(InputImg[imgsel]);
    split(InputcvImg,InputcvImgA);
}

//--------------------------------------------------------------
void ofApp::update(){


//pour actualiser le treshold du coup (affiche qqch seulement si "neck()" à déjà été lancée
    Neckcrop.cropFrom(InputImg[imgsel],0,Headtop+(hauteur/7)-halfneckH,InputImg[imgsel].getWidth()/*(3/2)*faces[0].width*/,3*halfneckH/*2*halfneckH*/);
    cv::cvtColor(toCv(Neckcrop),NeckcropMat,COLOR_BGRA2GRAY);
    cv::threshold(NeckcropMat,NeckcropMat,thresh,255,THRESH_BINARY);
    NeckcropctMat=NeckcropMat.clone();
    cv::findContours(NeckcropctMat,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

}

//--------------------------------------------------------------
void ofApp::draw(){

    ofSetColor(255,255,255);

    //draw image courrante
    InputImg[imgsel].draw(0,0);

    //draw pour le coup (affiche qqch seulement si "neck()" à déjà été lancée)
    drawMat(NeckcropMat,600,0);
    drawMat(NeckcropctMat,600,200);

    //affiche images qui servent dans le code de sonde pour debugger
    drawMat(outtest1,600,100);
    drawMat(outtest2,900,400);


        //drawMat(imginf,600,0);
        //drawMat(InputcvImgA[3],600,0);

        //cout<<(int) InputcvImgA[3].at<uchar>(250,250)<<endl;


    //affiche les lignes de proportion du personnage (H/7)
    ofDrawLine(0,Headtop,InputImg[imgsel].getWidth(),Headtop);
    ofDrawLine(0,Headtop+(hauteur/7),InputImg[imgsel].getWidth(),Headtop+(hauteur/7));
    ofDrawLine(0,Headtop+(2*hauteur/7),InputImg[imgsel].getWidth(),Headtop+(2*hauteur/7));
    ofDrawLine(0,Headtop+(3*hauteur/7),InputImg[imgsel].getWidth(),Headtop+(3*hauteur/7));
    ofDrawLine(0,Headtop+(4*hauteur/7),InputImg[imgsel].getWidth(),Headtop+(4*hauteur/7));
    ofDrawLine(0,Headtop+(5*hauteur/7),InputImg[imgsel].getWidth(),Headtop+(5*hauteur/7));
    ofDrawLine(0,Headtop+(6*hauteur/7),InputImg[imgsel].getWidth(),Headtop+(6*hauteur/7));
    ofDrawLine(0,Footfloor,InputImg[imgsel].getWidth(),Footfloor);

    ofSetColor(0,0,255);

    //affiche ligne de mi-hauteur du personnage
    ofDrawLine(0,Footfloor-(hauteur/2),InputImg[imgsel].getWidth(),Footfloor-(hauteur/2));
    ofSetColor(255,255,255);


    //affiche le cadre de la tête après détection
    ofNoFill();
    for(int i = 0; i < faces.size(); i++) {
        //Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        //ofDrawEllipse( faces[i].x + faces[i].width, faces[i].y + faces[i].height,faces[i].width, faces[i].height);
        ofDrawRectangle(faces[i].x,faces[i].y,faces[i].width,faces[i].height);
    }


    //draw GUI
    gui.draw();
}

void ofApp::removebkg(Mat src,int outputnumber)
{


        //1. Remove Shadows
        Mat hsvImg;
        cvtColor(src, hsvImg, CV_BGR2HSV); //converti en HSV
        Mat channel[3];
        split(hsvImg, channel);
        channel[2] = cv::Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);
        merge(channel, 3, hsvImg);
        Mat rgbImg;
        cvtColor(hsvImg, rgbImg, CV_HSV2BGR);
        //outtest1=gray.clone();

        //2. Convert to gray and normalize
        Mat gray(rgbImg.rows, src.cols, CV_8UC1);
        cvtColor(rgbImg, gray, CV_BGR2GRAY);
        cv::normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);
        //outtest2=gray.clone();

        //3. Edge detector
        cv::GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT);
        Mat edges;
        bool useCanny = false;
        if(useCanny){
            Mat detected_edges;
            int edgeThresh = 1;
            int lowThreshold = 250;
            int highThreshold = 750;
            int kernel_size = 5;
            cv::Canny(gray, detected_edges, lowThreshold, highThreshold, kernel_size);
            edges = detected_edges.clone();

        } else {
            //Use Sobel filter and thresholding.
            Mat edges1;
            //int scale = 1;
            int delta = 0;
            int ddepth = CV_16S;
            Mat edges_x, edges_y;
            Mat abs_edges_x, abs_edges_y;
            cv::Sobel(gray, edges_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
            cv::convertScaleAbs( edges_x, abs_edges_x );
            cv::Sobel(gray, edges_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
            cv::convertScaleAbs(edges_y, abs_edges_y);
            cv::addWeighted(abs_edges_x, 0.5, abs_edges_y, 0.5, 0, edges1);
            edges = edges1.clone();
            //Automatic thresholding
            //threshold(edges, edges, 0, 255, cv::THRESH_OTSU);
            //Manual thresholding
            cv::threshold(edges, edges, Threshold, 255, cv::THRESH_BINARY);
        }
        //outtest2=edges.clone();

        //4. Dilate
        Mat dilateGrad = edges.clone();
        int dilateType = MORPH_ELLIPSE;
        //dilateSize = 3;
        Mat elementDilate = getStructuringElement(dilateType,Size(2*dilateSize + 1, 2*dilateSize+1),Point(dilateSize, dilateSize));
        cv::dilate(edges, dilateGrad, elementDilate);

        outtest2=dilateGrad.clone();

        //5. Floodfill
        Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
        cv::floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0,cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
        floodFilled = cv::Scalar::all(255) - floodFilled;
        Mat temp;
        floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
        floodFilled = temp;
        //outtest2=floodFilled.clone();

        //6. Erode
        int erosionType = MORPH_ELLIPSE;
        //int erosionSize = 4;
        cv::Mat erosionElement = getStructuringElement(erosionType,
            cv::Size(2*erosionSize+1, 2*erosionSize+1),
            cv::Point(erosionSize, erosionSize));
        cv::erode(floodFilled, floodFilled, erosionElement);
        //outtest1=floodFilled.clone();

        //7. Find largest contour
        int largestArea = 0;
        int largestContourIndex = 0;
        Rect boundingRectangle;
        Mat largestContour(src.rows, src.cols, CV_8UC1, Scalar::all(0));
        cv::vector<vector<Point>> contours;
        cv::vector<Vec4i> hierarchy;
        cv::findContours(floodFilled, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        for(int i=0; i<contours.size(); i++)
        {
            double a = cv::contourArea(contours[i], false);
            if(a > largestArea)
            {
                largestArea = a;
                largestContourIndex = i;
                boundingRectangle = boundingRect(contours[i]);
            }
        }
        cv::Scalar color(255, 255, 255);
        cv::drawContours(largestContour, contours, largestContourIndex, color, CV_FILLED, 8, hierarchy); //Draw the largest contour using previously stored index.
        cv::rectangle(src, boundingRectangle, Scalar(0, 255, 0), 1, 8, 0);
        //outtest2=largestContour.clone();

        //8. Mask original image
        cv::Mat maskedSrc;
        src.copyTo(maskedSrc, largestContour);

        outtest1=maskedSrc.clone();

        //ajoute le masque dans l'alpha de l'image de sortie
        Mat imagenoBkgch[4];
        Mat matnoBkg;
        cv::split(maskedSrc, imagenoBkgch);
        largestContour.copyTo(imagenoBkgch[3]);
        cv::merge(imagenoBkgch,4,matnoBkg);

        //sauver l'image en png avec le numéro outputnumber
        ofImage imagenoBkg;
        toOf(matnoBkg,imagenoBkg);
        string pathfileout="nobkg/"+std::to_string(outputnumber)+"nobkg.png";
        imagenoBkg.save(pathfileout,OF_IMAGE_QUALITY_BEST);



}

void ofApp::neck(){
    //demi hauteur du crop pour le cou
    halfneckH=hauteur/60;
    //crop le la partie "cou"
    Neckcrop.cropFrom(InputImg[imgsel],0,Headtop+(hauteur/7)-halfneckH,InputImg[imgsel].getWidth()/*(3/2)*faces[0].width*/,3*halfneckH/*2*halfneckH*/);
    //threshold en fonction de la valuer choisie dans le gui
    cv::cvtColor(toCv(Neckcrop),NeckcropMat,COLOR_BGRA2GRAY);
    cv::threshold(NeckcropMat,NeckcropMat,thresh,255,THRESH_BINARY);

}

void ofApp::assemble(int jambe,int tronc)
{
    Mat Tronc;
    Mat Troncch[4];
    Mat Jambes;
    Mat Jambesch[4];
//lire contenu jambes
    dirjambes.listDir("jambes/");
//choisir au hazard (pour l'instant choisir avec les arguments)
    ofImage ofjambes;
    ofjambes.load("jambes/" + /*std::to_string(ofRandom(0,dirjambes.size()-1))*/std::to_string(jambe)+".png");
    Jambes=toCv(ofjambes);
    split(Jambes,Jambesch);
//lire troncs
    ofDirectory dirtroncs("troncs/");
    dirtroncs.listDir();
//choisir au hazard (pour l'instant choisir avec les arguments)
    ofImage oftronc;
    oftronc.load("troncs/" + /*std::to_string(ofRandom(0,dirtroncs.size()-1))*/std::to_string(tronc)+".png");
    Tronc=toCv(oftronc);
    split(Tronc,Troncch);

//mesurer jambes
    //trouver 1er pix de la taille (coté jambes) à gauche
        int i=0;
        int j=0;
        bool found=FALSE;

        outtest1=Jambesch[3].clone();

        //parcourir les pixels de la couche alpha pour trouver le 1er non nul
        while (found!=TRUE)
        {
            if ((int) Jambesch[3].at<uchar>(i,j)!=0)
            {found=TRUE;
             taillegauche=j;
            cout<<"taillegauche:"<<j<<endl;
            }
            else
            {
                j++;
                if(j>=Jambesch[3].cols)
                {
                   break;
                }
            }
        }

    //trouver 1er pix de la taille à droite
        i=0;
        j=Jambesch[3].cols-1;
        found=FALSE;
        while (found!=TRUE)
        {
            if ((int) Jambesch[3].at<uchar>(i,j)!=0)
            {found=TRUE;
             tailledroite=j;
            cout<<"tailledroite:"<<j<<endl;
            }
            else
            {
                j--;
                if(j>=Jambesch[3].cols)
                {
                    break;
                }
            }


        }

        largeurtaille=tailledroite-taillegauche;
        poscentretaille=taillegauche+largeurtaille/2;
         cout<<"largeurjambes :"<< largeurtaille << endl;

//mesurer tronc
    //coté gauche du tronc
        i=Troncch[3].rows-1;
        j=0;
        found=FALSE;
        while (found!=TRUE)
        {
            if ((int) Troncch[3].at<uchar>(i,j)!=0)
            {found=TRUE;
             troncgauche=j;
            cout<<"troncgauche:"<<j<<endl;
            }
            else
            {
                j++;
                if(j>=Troncch[3].cols)
                {
                   break;
                }
            }
        }

    //coté droit du tronc
        i=Troncch[3].rows-1;
        j=Troncch[3].cols-1;
        found=FALSE;
        while (found!=TRUE)
        {
            if ((int) Troncch[3].at<uchar>(i,j)!=0)
            {found=TRUE;
             troncdroit=j;
            cout<<"troncdroit:"<<j<<endl;
            }
            else
            {
                j--;
                if(j>=Troncch[3].cols)
                {
                    break;
                }
            }


        }

        largeurtronc=troncdroit-troncgauche;
        cout<<"largeurtronc :"<< largeurtronc << endl;
        //trouver ratio
        //ratiotroncjambes=largeurtronc/largeurjambes;


        outtest2=Jambes.clone();


//adapter l'un à l'autre en déformant les jambes. ou on pourrait choisir pour garder les meilleurs résultats
        //jambes adaptées
        Mat Jambesmod;
        // Matrice de transformation
        Mat lambda;
        // la mettre a 0
        lambda = Mat::zeros( Jambes.rows,Jambes.cols, Jambes.type() );

            //les pos des 4 points (rectangle clockwise) sur l'image d'entrés
            std::vector<cv::Point2f> inputQuad;
            inputQuad.push_back(cv::Point2f( taillegauche,0));
            inputQuad.push_back(cv::Point2f( tailledroite,0));
            inputQuad.push_back(cv::Point2f( Tronc.cols-1,Tronc.rows-1));
            inputQuad.push_back(cv::Point2f( 0,Tronc.rows-1  ));
            // les pos des 4 points sur l'image de sortie
            std::vector<cv::Point2f> outputQuad;
            outputQuad.push_back(cv::Point2f( troncgauche,0 ));
            outputQuad.push_back(cv::Point2f( troncdroit,0));
            outputQuad.push_back(cv::Point2f( Tronc.cols-1,Tronc.rows-1));
            outputQuad.push_back(cv::Point2f( 0,Tronc.rows-1  ));

            // trouve la transformation entre les deux rectangles
            lambda = cv::getPerspectiveTransform( inputQuad, outputQuad );

            // appliquer la transformation
            cv::warpPerspective(Jambes,Jambesmod,lambda,Jambesmod.size());

    outtest1=Jambesmod.clone();

//coller les deux images
    //créer grande image de sortie
    Mat resultmat(Tronc.rows+Jambesmod.rows,Jambesmod.cols,Tronc.type());
    resultmat.setTo(0);
    //copier le tronc dedans
    Tronc.copyTo(resultmat(Rect(0, 0, Tronc.cols, Tronc.rows)));
        cout<<"ligne"<<Tronc.rows<<"cols"<<Tronc.cols<<endl;
        cout<<"ligne"<<Jambesmod.rows<<"cols"<<Jambesmod.cols<<endl;
    //copier les jambes déformées dedans
    Jambesmod.copyTo(resultmat(Rect(0, Tronc.rows, Jambesmod.cols, Jambesmod.rows)));

//sauver l'image
    ofImage resultat;
    toOf(resultmat,resultat);
    string pathfileout="results/res.png";
    resultat.save(pathfileout,OF_IMAGE_QUALITY_BEST);

}

void ofApp::detectface()
{
    face_cascade.detectMultiScale(InputcvImg, faces,1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30), Size(200,200));//, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10) );
    cout<<"found"<<faces.size()<<endl;
}

void ofApp::cutjambes(){
    //cadre de coupe
    cv::Rect cadreinf(0,Headtop+hauteur/2,InputImg[imgsel].getWidth(),hauteur/2);
    //découpe
    imginf=InputcvImg(cadreinf);
    //sauve
    ofImage imageinf;
    toOf(imginf,imageinf);
    string pathfileout="jambes/"+std::to_string(imgsel)+".png";
    imageinf.save(pathfileout,OF_IMAGE_QUALITY_BEST);

}

void ofApp::cuthaut(){
    //cadre de coupe
    cv::Rect cadresup(0,Headtop,InputImg[imgsel].getWidth(),hauteur/2);
    //decoupe
    imgsup=InputcvImg(cadresup);
    //sauve
    ofImage imagesup;
    toOf(imgsup,imagesup);
    string pathfileout="troncs/"+std::to_string(imgsel)+".png";
    imagesup.save(pathfileout,OF_IMAGE_QUALITY_BEST);
}

void ofApp::findtopdown()
{
bool found;

int i=0;
int j=0;

//Scan pour trouver le haut de la tete
while (found!=TRUE)
{
    if ((int) InputcvImgA[3].at<uchar>(i,j)!=0)
    {found=TRUE;
     Headtop=i;
   // cout<<"haut:"<<i<<endl;
    }
    else
    {
        j++;
        if(j>=InputcvImgA[3].cols)
        {
            i++;
            j=0;
        }
    }


}

    i=(InputcvImgA[3].rows)-1;
    j=0;
    found=FALSE;

//Scan pour trouver le bas des pieds
    while (found!=TRUE)
    {
        if ((int) InputcvImgA[3].at<uchar>(i,j)!=0)
        {found=TRUE;Footfloor=i;
           // cout<<"bas:"<<i<<endl;
        }
        else
        {
            j++;
            if(j>=InputcvImgA[3].cols)
            {
                i--;
                j=0;
            }
        }

    }
found=FALSE;


//hauteur personnage
hauteur=Footfloor-Headtop;
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if(key=='c')
    {
        cutjambes();
        cuthaut();
    }
    if(key=='n')
    {
        neck();
    }
    if(key=='f')
    {
    findtopdown();
    detectface();
    }
    if(key=='r')
    {
    assemble(2,0); //assemble jambe et tronc par numéro de fichier
    //assemble(2,1);
    //assemble(0,2);
    //assemble(1,2);
    }
    if(key=='b')
    {
    removebkg(ofxCv::toCv(InputImg[imgsel]),imgsel);
    }

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

    //change l'image courante
    if (imgsel>=3)
    {imgsel=0;}
    else
    {imgsel++;}
    InputcvImg=toCv(InputImg[imgsel]);
    split(InputcvImg,InputcvImgA);

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
