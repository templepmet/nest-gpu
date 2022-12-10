#include <bits/stdc++.h>

using namespace std;

using ll = long long;
using P = tuple<int, int>;

const int MOD = 1e9 + 7;
const int INF = 1e9 + 1;

template <class T>
bool chmax(T &a, T b)
{
	if (a < b)
	{
		a = b;
		return true;
	}
	return false;
}
template <class T>
bool chmin(T &a, T b)
{
	if (a > b)
	{
		a = b;
		return true;
	}
	return false;
}

int main()
{
	int n = 1e8;

	int *a = new int[n];

	memset(a, 0, sizeof(int) * n);
	// fill(a, a + n, -1);

	delete a;

	return 0;
}