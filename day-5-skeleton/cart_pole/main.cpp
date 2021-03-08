#include "smarties.h"
#include "cart-pole.hpp"

#include <iostream>
#include <cstdio>


inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
{
  const int action_vars  = 1; // Force along x
  
  const int state_vars   = 1; // TODO: adjust
  comm->setStateActionDims(state_vars, action_vars);

  bool bounded = true;
  std::vector<double> upper_action_bound{  99.9 }; // TODO: adjust
  std::vector<double> lower_action_bound{ -99.9 }; // TODO: adjust
  comm->setActionScales(upper_action_bound, lower_action_bound, bounded);

  std::vector<bool> b_observable = { true }; //TODO: adjust
  comm->setStateObservable(b_observable);

  std::vector<double> upper_state_bound{ -99.9 }; // TODO: adjust
  std::vector<double> lower_state_bound{ +99.9 }; // TODO: adjust
  comm->setStateScales(upper_state_bound, lower_state_bound);

  CartPole env;

  while(true) //train loop
  {
    env.reset(comm->getPRNG());
    comm->sendInitState(env.getState());

    while (true) //simulation loop
    {
      std::vector<double> action = comm->recvAction(); // receive action from SMARTIES
      if(comm->terminateTraining()) return; // early exit

      // TODO: advance environment & and set 'terminated' flag
      //
      //

      bool terminated = true; // dummy

      // TODO: retrieve state from environment
      //
      //
      
      std::vector<double> state(state_vars); // dummy
      
      // TODO: retrieve reward from environment
      //
      //
      
      double reward = 99.9; // dummy

      if(terminated) 
      { 
        //tell smarties that this is a terminal state
        comm->sendTermState(state, reward);
        break;
      } 
      else 
      {
        comm->sendState(state, reward);
      }
    }
  }
}


int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main );
  return 0;
}
