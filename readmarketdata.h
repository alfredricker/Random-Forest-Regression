//THIS FILE INCLUDES SEVERAL RELEVANT FUNCTIONS TO RUN RANDOMFOREST.CPP
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <float.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <random>

using namespace std;

//calculate mean value of a vector
double calculateMean(const vector<double>& values) {
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    double size = (double)(values.size());
    return sum/size;
}

double calculateMSE(const vector<double>& actual, const vector<double>& predicted){
    double sum = 0.;
    if(actual.size()!=predicted.size()){
        cerr << "CalculateMSE error: vectors must be of same size" << endl;
        return -1;
    }
    for(int i = 0; i < actual.size(); i++){
        sum += (actual[i]-predicted[i])*(actual[i]-predicted[i]);
    }
    return sum/actual.size();
}


//calculate R-squared to measure performance of regression. must input the vector of dependent variables 
double calculateRsquared(const vector<double>&actual, const vector<double>&predicted){
    double sum1 = 0.;
    double sum2 = 0.;
    double ymean = calculateMean(actual);
    if(actual.size()!=predicted.size()){
        cerr << "CalculateRsquared error: vectors must be of same size" << endl;
        return -1;
    }
    if(actual.empty() || predicted.empty()){
        cerr << "CalculateRsquared error: empty vector" << endl;
        return -1;
    }
    //calculate sum squared regression (sum1) and sum of squares (sum2)
    for(int i = 0; i < actual.size(); i++){
        sum1 += (actual[i]-predicted[i])*(actual[i]-predicted[i]);
        sum2 += (actual[i]-ymean)*(actual[i]-ymean);
    }
    double rsquared = 1-(sum1/sum2);
    return rsquared;
}


// Function to calculate total error of a node's predictions
//this works for only one independent variable
double calculateTotalError(const vector<double>& actual, const vector<double>& predicted) {
    double totalError = 0.0;
    if(actual.size()!=predicted.size()){
        cerr << "CalculateTotalError error: vectors must be of same size" << endl;
        return -1;
    }
    for (int i=0; i<actual.size(); i++) {
        totalError += (actual[i] - predicted[i])*(actual[i]-predicted[i]);
    }
    return totalError;
}


//function that randomly samples n data sets from a vector of vectors. This is used to create the random forest
vector<vector<double>> randomSample(const vector<vector<double>>& data, int n){
    vector<vector<double>> sample;
    if(n > data.size()){
        cerr << "RandomSample error: sample size must be less than or equal to data size" << endl;
        return sample;
    }
    //randomly shuffle data
    vector<vector<double>> shuffledData = data;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(shuffledData.begin(), shuffledData.end(), default_random_engine(seed));
    //take first n data points
    for(int i = 0; i < n; i++){
        sample.push_back(shuffledData[i]);
    }
    return sample;
}


//reads data from a text file. dependent variable must be the last column for other functions to work properly.
vector<vector<string>> readDataFromFile(const string& filename) {
    vector<vector<string>> data;

    ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return data;
    }

    string line;
    while (getline(inputFile, line)) {
        vector<string> row;
        stringstream ss(line);
        string cell;

        while (ss >> cell) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    inputFile.close();
    return data;
}


void printDataFile(const vector<vector<string>>& fileData){
    for(const vector<string>& vec:fileData){
        for(string str : vec){
            cout << str << '\t';
        }
        cout << '\n';
    }
}


double oneWeekAverage(const vector<vector<string>>& data, int index){
    int i = 0;
    //can't calculate returns from first data point
    if(index == 0){
        index += 1;
    }
    double sum = 0.;
    double ret;
    while(i < 5 || index+i<data.size()){
        ret = (stod(data[index+i].back())+stod(data[index+i-1].back()))/stod(data[index+i-1].back());
        sum += ret;
        i++;
    }
    return ret/5.;
}


string findNextClosestDate(const std::string& targetDate, const vector<vector<std::string>>& vecdates) {
    // Convert the target date string to a time_point
    std::istringstream targetStream(targetDate);
    std::tm targetTm = {};
    targetStream >> get_time(&targetTm, "%Y-%m-%d");
    auto targetTime = std::chrono::system_clock::from_time_t(std::mktime(&targetTm));

    for (const vector<string>& dates : vecdates) {
            // Convert the current date string to a time_point
            std::istringstream currentStream(dates.front());
            std::tm currentTm = {};
            currentStream >> std::get_time(&currentTm, "%Y-%m-%d");
            auto currentTime = std::chrono::system_clock::from_time_t(std::mktime(&currentTm));

            if (currentTime == targetTime) {
                // Exact match found, return the data associated with the date
                return dates.front();
            }
            if (currentTime > targetTime) {
                // Next closest date found, return data associated with date
                return dates.front();
            }
        
    }

    // No matching or closest date found
    return "";
}

//target is target date to calculate returns
double calculateReturn(const string& target, const vector<vector<string>>& data){
    vector<string> previousData = data.front();
    for(const vector<string>& dat : data){
        if(dat.front()==target){
            return (stod(dat.back())-stod(previousData.back()))/stod(previousData.back());
        }
        previousData = dat;
    }
    //no matching returns
    return 0;
}


//this function just matches one independent variable with the average 1 week return following the data point 
//make sure that the dates are in the first column of text file as well as formatted properly (use textreformat.py)
vector<vector<double>> matchOneWeek(const vector<vector<string>>& indData, const vector<vector<string>>& depData){  
    vector<vector<double>> dat;
    //you could make these loops more efficient assuming the data files are ordered
    /*
    for(int i=0; i<indData.size(); i++){
        for(int d=0; d<depData.size(); d++){
            if(indData[i][0]==depData[d][0]){
                vector<double> newVec;
                //something goes wrong when you pass the data to the oneWeekAverage() function.
                newVec.push_back(stod(indData[i].back()));
                newVec.push_back(stod(depData[d].back()));
                dat.push_back(newVec);
            }
        }
    }*/
    for(const vector<string>& data: indData){
        vector<double> newVec;
        string dateData = findNextClosestDate(data.front(),depData);
        vector<double> depReturn;
        //calculate average returns for next 5 trading days
        for(int i=0; i<5; i++){
            depReturn.push_back(calculateReturn(dateData,depData));
            dateData = findNextClosestDate(dateData,depData);
            
        }
        double indReturn = calculateReturn(data.front(),indData);
        double oneWeekAverage = calculateMean(depReturn);
        newVec.push_back(indReturn);
        newVec.push_back(oneWeekAverage);
        //newVec.push_back(stod(data.back()));
        //newVec.push_back(stod(dateData));
        dat.push_back(newVec);
    }
    return dat;
}


//function to randomly shuffle vector of vectors
void shuffleVectorOfVectors(std::vector<std::vector<double>>& data) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(data.begin(), data.end(), generator);
}



//function to split the vector data and form training set and test set
template <typename T>
std::pair<std::vector<std::vector<T>>, std::vector<std::vector<T>>> formTrainingData(const std::vector<std::vector<T>>& inputVector) {
    std::size_t totalSize = inputVector.size();
    std::size_t splitIndex = static_cast<std::size_t>(totalSize * 0.8);

    // Shuffle the input vector randomly
    std::vector<std::vector<T>> shuffledVector = inputVector;
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(shuffledVector.begin(), shuffledVector.end(), generator);

    // Split the shuffled vector into two sets
    std::vector<std::vector<T>> first80(shuffledVector.begin(), shuffledVector.begin() + splitIndex);
    std::vector<std::vector<T>> last20(shuffledVector.begin() + splitIndex, shuffledVector.end());

    return std::make_pair(first80, last20);
}


template <typename T>
void printVector(const vector<T>& vec){
    for(const auto data: vec){
        cout<< data << '\t';
    }
}


void printDoubleVector(const vector<vector<double>>& doublevector){
    for(const vector<double>& data:doublevector){
        for(double dat: data){
            cout << dat << "\t";
        }
        cout << endl;
    }
}


//compares the first elements of vectors to sort vectors of vectors. this may be necessary for the fstep plot to plot properly
void sortVectorOfVectors(std::vector<std::vector<double>>& data) {
    std::sort(data.begin(), data.end(), [](const std::vector<double>& vec1, const std::vector<double>& vec2) {
        return vec1[0] < vec2[0];
    });
}

//function to read in data from second column of text file. format of the file should be date, price.
vector<vector<double>> readData(const string& filename){
    vector<vector<double>> data;
    ifstream file(filename);
    string line;
    double index = 0;
    while(getline(file,line)){
        vector<string> fileQueue;
        istringstream iss(line);
        string word;
        while(iss >> word){
            fileQueue.push_back(word);
        }
        double price = stod(fileQueue.back());
        data.push_back({index,price});
        index++;
    }
    return data;
}


void plotVectorOfVectors(const std::vector<std::vector<double>>& data,const string& filename) {
    // Create a temporary file to store the data
    std::string tempFileName = "temp.dat";
    std::ofstream tempFile(tempFileName);

    // Write the data to the temporary file
    for (const auto& vec : data) {
        tempFile << vec[0] << " " << vec[1] << "\n";
    }

    tempFile.close();

    // Open a pipe to communicate with Gnuplot
    FILE* gnuplotPipe = popen("gnuplot -persist", "w");
    if (!gnuplotPipe) {
        std::cerr << "Error: Failed to open the Gnuplot pipe." << std::endl;
        return;
    }

    // Send commands to Gnuplot through the pipe
    fprintf(gnuplotPipe, "set terminal pdfcairo\n");
    fprintf(gnuplotPipe, "set output '%s'\n",filename.c_str());
    fprintf(gnuplotPipe, "set title 'Linear Plot'\n");
    fprintf(gnuplotPipe, "set xlabel 'X'\n");
    fprintf(gnuplotPipe, "set ylabel 'Y'\n");
    fprintf(gnuplotPipe, "plot '%s' with fsteps\n",tempFileName.c_str());

    // Close the pipe
    pclose(gnuplotPipe);

    // Remove the temporary file
    remove(tempFileName.c_str());
}
