#pragma once

/*
 * This file must not be modified
*/


// helper struct to store states
struct Vec4
{
  double y1;
  double y2;
  double y3;
  double y4;

  Vec4(double _y1=0, double _y2=0, double _y3=0, double _y4=0) :
    y1(_y1), y2(_y2), y3(_y3), y4(_y4) {};

  Vec4 operator*(double v) const
  {
    return Vec4(y1*v, y2*v, y3*v, y4*v);
  }

  Vec4 operator+(const Vec4& v) const
  {
    return Vec4(y1+v.y1, y2+v.y2, y3+v.y3, y4+v.y4);
  }
};


// helper function to advance state
template <typename Func, typename Vec>
Vec rk46_nl(double dt, Vec u0, Func&& Diff)
{
  static constexpr double a[] = {0.000000000000, -0.737101392796, -1.634740794341, -0.744739003780, -1.469897351522, -2.813971388035};
  static constexpr double b[] = {0.032918605146,  0.823256998200, 0.381530948900,  0.200092213184,  1.718581042715,  0.270000000000};
  static constexpr int s = 6;
  Vec w;
  Vec u(u0);

  for (int i=0; i<s; ++i)
  {
    w = w*a[i] + Diff(u)*dt;
    u = u + w*b[i];
  }
  return u;
}




