#pragma once
#include <cmath>

template<class T>
class Vector3
{
public:
	constexpr Vector3() = default;
	constexpr Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

	constexpr T Length() { return std::sqrt(x * x + y * y + z * z); }
	constexpr static T Dot(Vector3<T> a, Vector3<T> b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

	constexpr Vector3<T> operator* (T scalar) { return Vector3 < T(x * scalar, y * scalar, z * scalar); }
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

	std::tuple<bool, Point3<T>, Point3<T>> Intersect(Ray<T> ray)
	{
		const auto L = ray.C - O;
		const auto tCa = Vector3<T>.Dot(L, D);

		if (tCa < 0)
			return std::make_tuple(false, Point3<T>(), Point3<T>());

		T d2 = Vector3<T>.Dot(L, L) - tCa * tCa;
		if (d2 > R * R)
			return std::make_tuple(false, Point3<T>(), Point3<T>());

		T thc = sqrt::(R * R - d2);
		T t0 = tca - thc;
		T t1 = tca + tch;

		auto p1 = ray.O + ray.D * t0;
		auto p2 = ray.O + ray.D * t1;
		return std::make_tuple(true, p1, p0);
	}

	Point3<T> C;
	T R = T(1);
};
