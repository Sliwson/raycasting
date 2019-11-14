#pragma once

template<class T>
class Vector3
{
public:
	Vector3() = default;
	Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

	T x = 0, y = 0, z = 0;
};

template<class T>
class Point3
{
public:
	Point3() = default;
	Point3(T x, T y, T z) : x(x), y(y), z(z) {}

	T x = 0, y = 0, z = 0;
};

template<class T>
class Ray
{
public:
	Ray() = default;
	Ray(Point3<T> origin, Vector3<T> direction) : O(origin), D(direction) {}
	
	Point3<T> O;
	Vector3<T> D;
};

template <class T>
class Sphere
{
	Sphere() = default;
	Sphere(Point3<T> center, T radius) : C(center), R(radius) {}

	Point3<T> C;
	T R = T(1);
};
