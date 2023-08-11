#include <vector>
#include <iostream>
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include "pointselector.hpp"
#include "readers.hpp"

using namespace cv;
using namespace std;
using namespace rapidjson;

void writePointPairs(const string& pointpairs_filename, const vector<PointPair>& pps) {
    Document pairs_doc(kArrayType);
    for(const PointPair& pp : pps) {
        Value left_point(kObjectType);
        left_point.AddMember("x", pp.points[0].x, pairs_doc.GetAllocator());
        left_point.AddMember("y", pp.points[0].y, pairs_doc.GetAllocator());
        Value right_point(kObjectType);
        right_point.AddMember("x", pp.points[1].x, pairs_doc.GetAllocator());
        right_point.AddMember("y", pp.points[1].y, pairs_doc.GetAllocator());

        Value pair(kObjectType);
        pair.AddMember("left", left_point, pairs_doc.GetAllocator());
        pair.AddMember("right", right_point, pairs_doc.GetAllocator());

        pairs_doc.PushBack(pair, pairs_doc.GetAllocator());
    }

    ofstream ofs(pointpairs_filename);
    OStreamWrapper osw(ofs);
 
    Writer<OStreamWrapper> writer(osw);
    pairs_doc.Accept(writer);
}

int main(int argc, char *argv[]) {
    Mat left = imread(argv[1]);
    Mat right = imread(argv[2]);

    unique_ptr<PointSelector> ps;
    if(argc > 3) {
        string pointpairs_filename(argv[3]);
        ps.reset(new PointSelector(readPointPairs(pointpairs_filename)));
    } else {
        ps.reset(new PointSelector());
    }
    vector<PointPair> pps = ps->select_points(left, right);
    writePointPairs("pointpairs.json", pps);
}
