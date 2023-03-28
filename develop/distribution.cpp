#include <bits/stdc++.h>

using namespace std;

#define NP 32

void convert(int *i_node, int *n_node, int p) {
    int np = NP;
    int i_assign_node = *i_node + ((p - (*i_node % np) + np) % np);
    if (i_assign_node >= *i_node + *n_node) {
        *i_node = -1;
        *n_node = 0;
    } else {
        *n_node = ((*i_node + *n_node - 1) - i_assign_node) / np + 1;
        *i_node = i_assign_node / np;
    }
}

// k * np + p = node_id

int main() {
    int n = 3;
    int a = 30;
    vector<int> m(NP, 0);
    for (int i = 0; i < NP; ++i) {
        int i_node = a;
        int n_node = n;
        convert(&i_node, &n_node, i);
        cout << "i=" << i << ":" << i_node << " " << n_node << endl;
        m[i] = n_node;
    }
    cout << "sum:" << accumulate(m.begin(), m.end(), 0) << endl;

    return 0;
}