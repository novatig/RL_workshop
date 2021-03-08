#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <functional>

#include "helpers.hpp"

struct CartPole
{
  
  // physical constants
  const double mp = 0.1;
  const double mc = 1;
  const double l  = 0.5;
  const double g  = 9.81;

  // time discretization 
  const double dt = 1e-4;
  const size_t simsteps_per_step = 100;

  // states & actions
  size_t step  = 0;   // current step
  double t     = 0.0; // current time
  double force = 0.0; // applied force 
  
  // vertical position of cart along x axis, volocity cart, angle between mass and vertical measured in radians, angular velocity of pole
  Vec4 u;

  // reset environment to random initial state
  void reset(std::mt19937& gen);

  // check if pole has fallen or left domain
  bool hasFailed();
  
  // check if simulation over
  bool isOver();

  // advance environment
  bool advance(const std::vector<double>& action);

  // get instanteneous reward
  double getReward();

  // helper function to calculate equations of motion
  Vec4 Diff(Vec4 _u);

  // retrieve state from environment
  std::vector<double> getState();

};


void CartPole::reset(std::mt19937& gen)
{
    // Creates random initial position.
	std::uniform_real_distribution<double> dist(-0.05,0.05);
	u = Vec4(dist(gen), dist(gen), dist(gen), dist(gen));
    step  = 0;
	force = 0;
    t     = 0;
}


bool CartPole::hasFailed()
{
   return std::fabs(u.y1)>2.4 || std::fabs(u.y3)>M_PI/15;
}
 

bool CartPole::isOver()
{
   return step>=500 || hasFailed();
}


double CartPole::getReward()
{
    if ( hasFailed() )  { return 0; }
    else                { return 1; }
}


bool CartPole::advance(const std::vector<double>& action)
{
    force = action[0];
    step++;
    for (size_t i=0; i<simsteps_per_step; i++) 
    {
      u = rk46_nl(dt, u, std::bind(&CartPole::Diff, this, std::placeholders::_1));
      t += dt;
      if( isOver() ) return true;
    }
    return false;
}


Vec4 CartPole::Diff(Vec4 _u)
{
    Vec4 res;

    const double cosy = std::cos(_u.y3);
    const double siny = std::sin(_u.y3);
    const double w = _u.y4;
    const double totMass = mp+mc;
    const double fac2 = l*(4.0/3 - (mp*cosy*cosy)/totMass);
    const double F1 = force + mp * l * w * w * siny;
    res.y4 = (g*siny - F1*cosy/totMass)/fac2;
    res.y2 = (F1 - mp*l*res.y4*cosy)/totMass;
    res.y1 = _u.y2;
    res.y3 = _u.y4;

    return res;
}


std::vector<double> CartPole::getState()
{
    std::vector<double> state;

    // TODO: populate vector and return it
    //
    //
	
	return state;
}
  


