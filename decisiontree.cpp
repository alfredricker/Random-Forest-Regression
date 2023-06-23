#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <queue>
#include <functional>
#include "readmarketdata.h"

using namespace std;

// Structure to represent a node in the decision tree
struct Node {
    double predictedValue;          // The predicted value at this node
    int featureIndex;      // Index of the feature used for splitting
    double splitValue;     // The threshold value for splitting
    vector<vector<double>> dataSet; //The dataset that belongs to this node
    Node* left;            // Pointer to the left child node
    Node* right;           // Pointer to the right child node
    Node* parent;
    Node(int val): predictedValue(static_cast<double>(val)), featureIndex(val), splitValue(static_cast<double>(val)), dataSet({{}}), left(nullptr), right(nullptr), parent(nullptr) {}
};

struct Split {
    int featureIndex;
    double splitValue;
};


double squareDifference(const vector<double>& values){
    double mean = calculateMean(values);
    double sum = 0.;
    for (double val : values){
        double sqdiff = val - mean;
        sum += (sqdiff*sqdiff);
    }
    return sum;
}

double calculateVariance(const vector<double>& values){
    if(values.size()<=1){
        cerr<< "calculateVariance error: vector must have size greater than 1" << endl;
        return -1.;
    }
    double mean = calculateMean(values);
    double sum = 0.;
    for (double val : values){
        double diff = val - mean;
        sum += (diff*diff);
    }
    return sum/(values.size()-1);
}



//returns the vector of dependent variables (which should be the last entry of each vector)
vector<double> callTargets(const vector<vector<double>>& dataset){
    vector<double> dep;
    for (const vector<double>& data : dataset){
        dep.push_back(data.back());
    }
    return dep;
}


// Function to compute the total number of leaf nodes in the decision tree. this is useful for cost complexity pruning
//must input an integer set to zero in the main function.
void countLeafNodes(Node* node, int& num) {
    if (node->right == nullptr && node->right == nullptr) {
        num += 1;
    }
    else{
        countLeafNodes(node->left,num);
        countLeafNodes(node->right,num);
    }
}


// Function to split the dataset based on a given feature and split value
pair<vector<vector<double>>, vector<vector<double>>> splitDataset(const vector<vector<double>>& dataset, int featureIndex, double splitValue) {
    vector<vector<double>> leftSubset;
    vector<vector<double>> rightSubset;
    
    for (const vector<double>& dataPoint : dataset) {
        if (dataPoint[featureIndex] <= splitValue) {
            leftSubset.push_back(dataPoint);
        } 
        else {
            rightSubset.push_back(dataPoint);
        }
    }
    
    return make_pair(leftSubset, rightSubset);
}


void printLeafNodes(Node* node) {
    if (node == nullptr) {
        return;
    }

    if (node->left == nullptr && node->right == nullptr) {
        // Leaf node found, print its value
        printDoubleVector(node->dataSet);
        std::cout << "predicted value: " << node->predictedValue << "\n\n";
    }

    // Recursively check the left and right subtrees
    printLeafNodes(node->left);
    printLeafNodes(node->right);
}


//returns a vector containing all of the leaf nodes of the decision tree. this is useful for cost complexity pruning
vector<Node*> getLeafNodes(Node* root){
    vector<Node*> nodeVector;
    queue<Node*> nodeQueue;
    nodeQueue.push(root);
    while(!nodeQueue.empty()){
        //is a leaf node
        if(nodeQueue.front()->right==nullptr&&nodeQueue.front()->left==nullptr){
            nodeVector.push_back(nodeQueue.front());
        }
        else{
            nodeQueue.push(nodeQueue.front()->left);
            nodeQueue.push(nodeQueue.front()->right);
        }
        nodeQueue.pop();
    }
    return nodeVector;
}


//prints the pair of vectors returned in the data split. this function was just used for testing purposes.
void printSplit(const pair<vector<vector<double>>,vector<vector<double>>>& split){
    for (const vector<double> &data : split.first){
        for (const double &values : data){
            cout << values << " ";
        }
        cout << endl;
    }
    cout << endl;
    for (const vector<double> &data: split.second){
        for (const double &values : data){
            cout << values << " ";
        }
        cout << endl;
    }
}


//calculate best data split as the one that minimizes the MSE
Split* bestsplit(const vector<vector<double>>& dataset){
    double min = 1e10;

    int bestIndex = 0;
    double bestSplit = 0.;
    Split* dataSplit = new Split;
    dataSplit->featureIndex = bestIndex;
    dataSplit->splitValue= bestSplit;

    //gets the number of data features
    int dataSize = dataset[0].size();
    for (const vector<double> &data : dataset){
        if(data.size() != dataSize){
            cout << "Error: data must all be of the same dimension!" << endl;
        }
    }
    //loop through all independent values
    vector<vector<double>> columns(dataSize-1);
    for (int i = 0; i<dataSize-1; i++){
        for (const vector<double> &data : dataset){
            columns[i].push_back(data[i]);
        }
    }

    //loop through each independent variable of the data to find the best split
    for (int f = 0; f<dataSize-1; f++){
        for (int i = 0; i< dataset.size()-1; ++i){
            pair<vector<vector<double>>, vector<vector<double>>> newSplit = splitDataset(dataset, f, dataset[i][f]);
            double sumMSE = squareDifference(callTargets(newSplit.first)) + squareDifference(callTargets(newSplit.second)); 
            if(sumMSE < min){
                dataSplit->featureIndex = f;
                dataSplit->splitValue= dataset[i][f];
                min = sumMSE;
            }            
        }
    }
    return dataSplit;
}


//the data vector should be the vector of independent variables that are used to make a prediction
double predict(Node* node, const vector<double>& data) {
    if (node->left == nullptr && node->right == nullptr) {
        return node->predictedValue; // Leaf node, return the predicted value
    }

    if (data[node->featureIndex] <= node->splitValue) {
        return predict(node->left, data); // Traverse left
    } else {
        return predict(node->right, data); // Traverse right
    }
}


//dataset is the training data
Node* decisionTree(const vector<vector<double>>& dataset){    
    Node* rootNode = new Node(0);
    rootNode->dataSet = dataset;
    int dataSize = dataset.size();
    int proportion = dataSize/20;

    int currentdepth = 1;
    //will loop through the nodes at each depth by using a queue
    queue<Node*> nodeQueue;
    nodeQueue.push(rootNode);

    //loop through all depths
    while(!nodeQueue.empty()){
        //loop through all branches at the current depth
        vector<vector<double>> data = nodeQueue.front()->dataSet;
        Split* split = new Split;
        split = bestsplit(data);
        // Split the dataset based on the best feature and split value
        pair<vector<vector<double>>, vector<vector<double>>> subsets = splitDataset(data, split->featureIndex, split->splitValue);
        double var = squareDifference(callTargets(subsets.first)) + squareDifference(callTargets(subsets.second));
        double varDiff = squareDifference(callTargets(data)) - var;
        
        //surprisingly this process directly modifies the values of the child nodes in the rootNode (which is what I want)
        if(data.size()<=proportion){ //second terminating condition
            //nodeQueue.front()->featureIndex = split->featureIndex;
            //nodeQueue.front()->splitValue = split->splitValue;
            nodeQueue.front()->predictedValue = calculateMean(callTargets(data));
            nodeQueue.front()->left = nullptr;
            nodeQueue.front()->right = nullptr; 
        }
        else{
            nodeQueue.front()->featureIndex = split->featureIndex;
            nodeQueue.front()->splitValue = split->splitValue;
            nodeQueue.front()->predictedValue = calculateMean(callTargets(data));
            Node* leftNode = new Node(0);
            Node* rightNode = new Node(0);
            leftNode->dataSet = subsets.first;
            rightNode->dataSet = subsets.second;
            leftNode->parent = nodeQueue.front();
            rightNode->parent = nodeQueue.front();
            nodeQueue.front()->left = leftNode;
            nodeQueue.front()->right = rightNode;
        }
        if(nodeQueue.front()->left!=nullptr && nodeQueue.front()->right!=nullptr){
            nodeQueue.push(nodeQueue.front()->left);
            nodeQueue.push(nodeQueue.front()->right);
        }
        nodeQueue.pop();
        delete split;
        }
          
    return rootNode;
}


// Function to perform cost complexity pruning of the decision tree using k-fold cross-validation
//Since the node structure is a pointer, the tree may be pruned without having to return a new tree
void costComplexityPruning(Node* root,const vector<vector<double>>&trainingData, double alpha){
    //Node* prunedTree = new Node(0); //why does this modify the root node?
    Node* prunedTree = new Node(*root);
    vector<Node*> leafNodes = getLeafNodes(prunedTree);
    double leafSize = leafNodes.size();
    double totalError = 0;
    for(const vector<double>& data : trainingData){
        double prediction = predict(prunedTree, data);
        double actual = data.back();
        totalError += pow((prediction-actual),2);
    }
    totalError += alpha*leafSize;

    //rather than loop through the leaf nodes, loop through the parent nodes of the leaf nodes
    //need to prune nodes that are of the form (x,y) where x and y are leaf nodes. in other words, you need to make sure that the parent node is of two leaf nodes.
    //loop through all leaf nodes
    bool pruned = true;
    while(pruned == true){
        pruned = false;
        for(Node* leaf: leafNodes){
            if(leaf->parent!=nullptr){
                Node* parent = leaf->parent;
                //if the parent node is of the form (x,y) where x and y are leaf nodes, then prune the parent node
                if(parent->left!=nullptr && parent->right!=nullptr){
                    //deep copy the parent node
                    Node* parentCopy = new Node(*parent);
                    if(parent->left->left==nullptr && parent->left->right==nullptr && parent->right->left==nullptr && parent->right->right==nullptr){
                        //prune the parent node
                        parent->left = nullptr;
                        parent->right = nullptr;
                        //calculate the error of the pruned tree
                        double prunedError = 0;
                        for(const vector<double>& data : trainingData){
                            double prediction = predict(prunedTree, data);
                            double actual = data.back();
                            prunedError += pow((prediction-actual),2);
                        }
                        prunedError += alpha*(leafSize-1);
                        //if the error of the pruned tree is less than the error of the unpruned tree, then keep the pruned tree
                        if(prunedError<totalError-0.5){
                            totalError = prunedError;
                            pruned = true;
                        }
                        else{
                            //if the error of the pruned tree is greater than the error of the unpruned tree, then keep the unpruned tree
                            parent = parentCopy;
                        }

                    }
                    delete parentCopy;
                }
            }
            else{
                cerr<<"leaf node has no parent"<<endl;
            }
        }
    }
}


// Function to perform k-fold cross-validation to get the best alpha value for cost complexity pruning
double kFoldCrossValidation(Node* root, const vector<vector<double>>& trainingData, int k){
    //split the training data into k folds
    vector<vector<vector<double>>> folds;
    int foldSize = trainingData.size()/k;
    //this doesn't work if the training data size is not divisible by k
    for(int i=0;i<k;i++){
        vector<vector<double>> fold;
        if(i!=k-1){
            for(int j=0;j<foldSize;j++){
                fold.push_back(trainingData[i*foldSize+j]);
            }
        }
        //if i = k-1, push back the remaining data
        else{
            for(int j=foldSize*k;j<trainingData.size();j++){
                fold.push_back(trainingData[j]);
            }
        }
        folds.push_back(fold);
    }
    
    //for each fold, train the tree on the remaining k-1 folds, prune the tree, and evaluate MSE using the remaining fold
    double bestAlpha = 0;
    double bestMSE = 1e10;
    for(double alpha=0;alpha<2;alpha+=0.2){
        double averageMSE = 0.;
        for(int i=0;i<k;i++){
            vector<vector<double>> trainingSet;
            for(int j=0;j<k;j++){
                if(j!=i){
                    for(const vector<double>& data : folds[j]){
                        trainingSet.push_back(data);
                    }
                }
            }
            //train the tree on the training set
            Node* trainedTree = decisionTree(trainingSet);
            //prune the tree using the remaining fold
            costComplexityPruning(trainedTree,trainingData,alpha);
            vector<double> predictions;
            vector<double> actuals;
            for(const vector<double>& data : folds[i]){
                double prediction = predict(trainedTree,data);
                double actual = data.back();
                predictions.push_back(prediction);
                actuals.push_back(actual);
            }
            averageMSE += calculateMSE(predictions,actuals);
        }
        averageMSE /= k;
        if(averageMSE<bestMSE){
            bestMSE = averageMSE;
            bestAlpha = alpha;
        }
    }
    return bestAlpha;
}



int main(int argc, char* argv[]) {
    //inputting the data that I want to examine the relationship between
    string eps = "data/ref_aapleps.txt";
    string aapl = "data/ref_aapl.txt";
    vector<vector<string>> epsData = readDataFromFile(eps);
    vector<vector<string>> aaplData = readDataFromFile(aapl);
    //this function does a lot of things. It computes return values of both eps and aapl price data, and matches the eps data with the average one week return
    vector<vector<double>> dataVector = matchOneWeek(epsData,aaplData);

    vector<vector<double>> aapldat = readData("data/ref_aapl.txt");

    //split our data into a training set and a test set
    pair<vector<vector<double>>,vector<vector<double>>> trainAndTest = formTrainingData(aapldat);
    vector<vector<double>> trainingData = trainAndTest.first;
    vector<vector<double>> testData = trainAndTest.second;

    //form the decision tree
    Node* decision = decisionTree(trainingData);
    double alpha = kFoldCrossValidation(decision, trainingData, 5);
    costComplexityPruning(decision, trainingData, alpha);

    //evaluate performance
    vector<double> predictedVals;
    vector<double> actualVals;
    //create vector of predicted values
    for(const vector<double> data: testData){
        double predictVal = predict(decision, data);
        predictedVals.push_back(predictVal);
        //cout << "Predicted: " << predictVal << " Actual: " << data.back() << endl;
        actualVals.push_back(data.back());
    }
    double rsquared = calculateRsquared(actualVals,predictedVals);
    cout << "R-Squared: " << rsquared << endl;


    //plotting the leaf nodes
    vector<vector<double>> leaf;
    for(Node* node: getLeafNodes(decision)){
        for(const vector<double> data: node->dataSet){
            leaf.push_back({data.front(),node->predictedValue});
        }
    }
    sortVectorOfVectors(leaf);
    plotVectorOfVectors(leaf,"aapleps.pdf");
    delete decision;

    return 0;
}