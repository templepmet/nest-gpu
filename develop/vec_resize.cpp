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
	vector<int> delay_hist(3, 0);

	std::string filename = "syndelay_" + std::to_string(2) + ".txt";
	std::ofstream ofs(filename);
	for (int i = 0; i < delay_hist.size(); ++i)
	{
		if (i > 0)
			ofs << ", ";
		ofs << delay_hist[i];
	}
	ofs.close();

	return 0;
}