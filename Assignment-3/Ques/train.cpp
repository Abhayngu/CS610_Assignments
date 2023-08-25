#include <iostream>

using namespace std;

#define fori(x, y) for(int i = x; i<y; i++)
#define forj(x, y) for(int j = x; j<y; j++)
#define fork(x, y) for(int k = x; k<y; k++)
#define forl(x, y) for(int l = x; l<y; l++)

int main(){
	int n = 6;
	int i, j, k, l;
	fori(0, n){
		forj(1, n){
			fork(j, i){
				forl(1, k){
					cout << i << " " << j << " " << k << " " << l << endl;
				}
			}
		}
	}
	cout << endl << endl;

	forj(1, n){
		fori(max(2, j), n){
			forl(1, i){
				fork(max(j, l+1), i){
					cout << i << " " << j << " " << k << " " << l << endl;
				}
			}
		}
	}
}