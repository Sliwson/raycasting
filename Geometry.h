#pragma once
#include <cmath>
#include <tuple>

constexpr auto PI = 3.14159265358979323846f;

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

	constexpr T Length() const { return std::sqrt(x * x + y * y + z * z); }
	constexpr static T Dot(Vector3<T> a, Vector3<T> b) const { return a.x * b.x + a.y * b.y + a.z * b.z; }

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

	std::tuple<bool, Point3<T>, Point3<T>> Intersect(Ray<T> ray) const
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

template<class T>
class Matrix4
{
public:
	Matrix4 = default();

	const T* operator [] (int i) const { return M[i]; }
	T* operator [] (int i) { return M[i]; }

	Matrix<T> operator* (const Matrix4<T> rhs) const
	{
		Matrix4 result;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				result[i][j] = M[i][0] * rhs[0][j] + M[i][1] * rhs[1][j] + M[i][2] * rhs[2][j] + M[i][3] * rhs[3][j];

		return result;
	}

	Point3<T> Translate(Point3<T> p)
	{
		auto point = Point3<T>(p.x * M[0][0] + p.y * M[1][0] + p.z * M[2][0] + M[3][0],
			p.x * M[0][1] + p.y * M[1][1] + p.z * M[2][1] + M[3][1],
			p.x * M[0][2] + p.y * M[1][2] + p.z * M[2][2] + M[3][2]);

		T w = p.x * M[0][3] + p.y * M[1][3] + p.z * M[2][3] + M[3][3];
		if (w != 1 && w != 0)
		{
			point.x /= w; point.y /= w; point.z /= w;
		}

		return point;
	}

	Vector3<T> Translate(Vector3<T> p)
	{
		return Vector3<T>(p.x * M[0][0] + p.y * M[1][0] + p.z * M[2][0],
			p.x * M[0][1] + p.y * M[1][1] + p.z * M[2][1],
			p.x * M[0][2] + p.y * M[1][2] + p.z * M[2][2]);
	}

private:

	T M[4][4] = { {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1} };
};

