#pragma once
#include <cmath>
#include <tuple>

template<class T>
class Point3
{
public:
	constexpr Point3() = default;
	constexpr Point3(T x, T y, T z) : x(x), y(y), z(z) {}

	constexpr Point3<T> operator+ (Point3<T> other) { return Point3<T>(x + other.x, y + other.y, z + other.z); }
	constexpr Point3<T> operator- (Point3<T> other) { return Point3<T>(x - other.x, y - other.y, z - other.y); }

	T x = 0, y = 0, z = 0;
};

template<class T>
class Vector3
{
public:
	constexpr Vector3() = default;
	constexpr Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
	constexpr Vector3(Point3<T> p) : x(p.x), y(p.y), z(p.z) {}

	constexpr T Length() { return std::sqrt(x * x + y * y + z * z); }
	constexpr static T Dot(Vector3<T> a, Vector3<T> b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

	constexpr Vector3<T> operator* (T scalar) { return Vector3 <T>(x * scalar, y * scalar, z * scalar); }
	constexpr Vector3<T> operator/ (T scalar) { return Vector3<T>(x / scalar, y / scalar, z / scalar); }

	void Normalize()
	{
		const auto length = Length();
		x /= length;
		y /= length;
		z /= length;
	}

	T x = 0, y = 0, z = 0;
};

template<class T>
Point3<T> operator+(Point3<T> lhs, Vector3<T> rhs)
{
	return Point3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

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
public:
	Sphere() = default;
	Sphere(Point3<T> center, T radius) : C(center), R(radius) {}

	std::tuple<bool, Point3<T>, Point3<T>> Intersect(Ray<T> ray)
	{
		const auto L = Vector3<T>(C - ray.O);
		auto tCa = Vector3<T>::Dot(L, ray.D);

		if (tCa < 0)
			return std::make_tuple(false, Point3<T>(), Point3<T>());

		T d2 = Vector3<T>::Dot(L, L) - tCa * tCa;
		if (d2 > R * R)
			return std::make_tuple(false, Point3<T>(), Point3<T>());

		T thc = std::sqrt(R * R - d2);
		T t0 = tCa - thc;
		T t1 = tCa + thc;

		auto p1 = ray.O + ray.D * t0;
		auto p2 = ray.O + ray.D * t1;
		return std::make_tuple(true, p1, p2);
	}

	Point3<T> C;
	T R = T(1);
};
