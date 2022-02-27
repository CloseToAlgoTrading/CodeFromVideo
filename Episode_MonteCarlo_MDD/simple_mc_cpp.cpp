//g++-11 -std=c++17 -O3 simple_mc_cpp.cpp -o simple_mc_cpp

#include <algorithm>    // Needed for the "max" function
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <numeric> 
#include <execution>

#include <bits/stdc++.h>
#include <chrono>

using namespace std;


auto MaxDrawdown(const vector<double>& stock)
{
    auto mdd = 0.0;
    auto maxPrev = stock.front();
    for (auto i=1u; i<stock.size(); ++i)
    {
        mdd = max(mdd, maxPrev - stock[i]);
        maxPrev = max(maxPrev, stock[i]);
    }
    return mdd;
}


// Mean DrawDown with a Monte Carlo method
double monte_carlo3(vector<double> dd, const double &S0, const double& mu, const double& sigma, const int &N, const int &N_PATH, const vector<double> noise)
{
    const double drift_0 = (mu - 0.5 * sigma*sigma);
    int idx = 0;

    for(int i = 0; i < N_PATH; ++i)
    {
        double S;
        auto mdd = 0.0;
        auto maxPrev = S0;
        double W = 0.0;

        for (int k=1; k<=N; ++k) 
        {
            idx = (k-1) + (i) * N;
            W += noise[idx];
            S = S0*exp((drift_0 * k) + (sigma * W));
            mdd = max(mdd, (maxPrev - S)/maxPrev);
            maxPrev = max(maxPrev, S);
        }
        dd[i] = mdd;
    }

  return reduce(dd.begin(), dd.end()) / dd.size();
}

int main(int argc, char **argv) {

    const double S0 = 205.42999267578125;
    const double mu = 0.00035128687077953227;
    const double sigma = 0.00781510977845114;
    const double N_PATH = 10000;
    const double N = 100;    
    
    std::random_device rd{};
    std::mt19937 gen{rd()};

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<double> d{0.0f,1.00f};

    vector<double> dd(N_PATH);


    const size_t N_NORMALS = (size_t)N * N_PATH;
    vector<double> noise(N_NORMALS);
    generate(noise.begin(), noise.end(), [&] () { return d(gen); });

   auto start = chrono::high_resolution_clock::now(); 
   ios_base::sync_with_stdio(false);
   
   auto mdd = monte_carlo3(dd, S0, mu, sigma, N, N_PATH, noise);
   //this_thread::sleep_for(std::chrono::seconds(1));

   auto end = chrono::high_resolution_clock::now();
   double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
   time_taken *= 1e-9;
   printf("execution time: %f s\nmdd: %f %% \n", time_taken, mdd*100);

  return 0;
}
