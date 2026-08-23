#include <iostream>
#include <vector>
using namespace std;

class Tensor {
    public:
    Tensor ( const vector < size_t >& shape ,const vector < double >& values ){}
};

int main() {
    Tensor A = Tensor :: zeros ({2 , 3}) ;
    Tensor B = Tensor :: ones ({3 , 3}) ;
    Tensor C = Tensor :: random ({2 , 2} , 0.0 , 1.0) ;
    Tensor D = Tensor :: arange (0 , 6);

};
