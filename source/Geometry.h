#pragma once
#include <cmath>
#include <tuple>

constexpr auto PI = 3.14159265358979323846f;

namespace Geometry {

	template<class T>
	class Point3
	{
	public:
		constexpr Point3() = default;
		__device__ __host__	constexpr Point3(T x, T y, T z) : x(x), y(y), z(z) {}

		__device__ __host__ constexpr Point3<T> operator+ (const Point3<T>& other) const { return Point3<T>(x + other.x, y + other.y, z + other.z); }
		__device__ __host__ constexpr Point3<T> operator- (const Point3<T>& other) const { return Point3<T>(x - other.x, y - other.y, z - other.z); }

		T x = 0, y = 0, z = 0;
	};

	template<class T>
	class Vector3
	{
	public:
		constexpr Vector3() = default;
		__device__ __host__ constexpr Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
		__device__ __host__	constexpr Vector3(Point3<T> p) : x(p.x), y(p.y), z(p.z) {}

		__device__ __host__	constexpr T Length() const { return std::sqrt(x * x + y * y + z * z); }
		__device__ __host__ constexpr T LengthSquared() const { return x * x + y * y + z * z; }
		__device__ __host__	constexpr static T Dot(const Vector3<T>& a, const Vector3<T>& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

		__device__ __host__	constexpr Vector3<T> operator* (T scalar) const { return Vector3 <T>(x * scalar, y * scalar, z * scalar); }
		__device__ __host__	constexpr Vector3<T> operator/ (T scalar) const { return Vector3<T>(x / scalar, y / scalar, z / scalar); }
		__device__ __host__ constexpr Vector3<T> operator- (const Vector3<T>& other) { return Vector3<T>(x - other.x, y - other.y, z - other.z); }

		__device__ __host__	void Normalize()
		{
			const auto length = Length();
			x /= length;
			y /= length;
			z /= length;
		}

		T x = 0, y = 0, z = 0;
	};

	template<class T>
	__device__ __host__ Point3<T> inline operator+(const Point3<T>& lhs, const Vector3<T>& rhs)
	{
		return Point3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
	}

	template<class T>
	class Ray
	{
	public:
		Ray() = default;
		__device__ __host__	Ray(Point3<T> origin, Vector3<T> direction) : O(origin), D(direction) {}

		Point3<T> O;
		Vector3<T> D;
	};

	template <class T>
	class Sphere
	{
	public:
		Sphere() = default;
		__device__ __host__	Sphere(Point3<T> center, T radius) : C(center), R(radius) {}

		__device__ __host__	bool Intersect(const Ray<T>& ray, Point3<T>* p) const
		{
			const auto L = Vector3<T>(C - ray.O);
			auto tCa = Vector3<T>::Dot(L, ray.D);

			if (tCa < 0)
				return false;

			T d2 = Vector3<T>::Dot(L, L) - tCa * tCa;
			if (d2 > R* R)
				return false;

			T thc = std::sqrt(R * R - d2);
			T t0 = tCa - thc;
			T t1 = tCa + thc;

			if (t0 > t1)
			{
				T tmp = t0;
				t0 = t1;
				t1 = tmp;
			}

			if (t0 < 0)
			{
				t0 = t1;
				if (t0 < 0)
					return false;
			}

			*p = ray.O + ray.D * t0;
			return true;
		}

		Point3<T> C;
		T R = T(1);
	};

	template<class T>
	class Matrix4
	{
	public:
		Matrix4<T>() = default;

		__device__ __host__	const T* operator [] (int i) const { return M[i]; }
		__device__ __host__	T* operator [] (int i) { return M[i]; }

		__device__ __host__	Matrix4<T> operator* (const Matrix4<T> rhs) const
		{
			Matrix4 result;
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					result[i][j] = M[i][0] * rhs[0][j] + M[i][1] * rhs[1][j] + M[i][2] * rhs[2][j] + M[i][3] * rhs[3][j];

			return result;
		}

		__device__ __host__	Point3<T> Translate(Point3<T> p)
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

		__device__ __host__	Vector3<T> Translate(Vector3<T> p)
		{
			return Vector3<T>(p.x * M[0][0] + p.y * M[1][0] + p.z * M[2][0],
				p.x * M[0][1] + p.y * M[1][1] + p.z * M[2][1],
				p.x * M[0][2] + p.y * M[1][2] + p.z * M[2][2]);
		}

	private:

		T M[4][4] = { {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1} };
	};

	// class-code
	float3 GetColorHost(const int imageW, const int imageH, const int x, const int y)
	{
		//calculate camera ray
		Matrix4<float> cameraToWorld;
		float scale = tan(PI / 4);
		float imageAspectRatio = imageW / (float)imageH;
		auto origin = cameraToWorld.Translate(Point3<float>());
		float rayx = (2 * ((float)x + 0.5f) / (float)imageW - 1) * scale * imageAspectRatio;
		float rayy = (1 - 2 * ((float)y + 0.5f) / (float)imageH) * scale;
		auto direction = cameraToWorld.Translate(Vector3<float>(rayx, rayy, -1));
		direction.Normalize();
		auto cameraRay = Ray<float>(origin, direction);

		//hardcoded constants
		auto light = Point3<float>(-200, -200, -200);
		float3 color = { 110.f / 255, 193.f / 255, 248.f / 255 };
		auto sphere = Sphere<float>(Point3<float>(0.f, imageH / 10.f, -4.f), imageH / 10.f);
		float3 sphereColor = { .9f, .9f, 0.f };
		float kd = 0.5;
		float ks = 0.5;
		int alpha = 10;

		//intersection
		Point3<float> intersection;
		auto result = sphere.Intersect(cameraRay, &intersection);
		if (result)
		{
			auto normal = Vector3<float>(intersection - sphere.C);
			normal.Normalize();
			auto lightVector = Vector3<float>(light - intersection);
			lightVector.Normalize();

			auto r = normal * 2.f * Vector3<float>::Dot(lightVector, normal) - lightVector;
			auto view = Vector3<float>(intersection - origin);
			view.Normalize();

			float kdm = kd * Vector3<float>::Dot(normal, lightVector);
			float ksm = ks * pow(Vector3<float>::Dot(r, view), alpha);
			float multiplier = kdm + ksm;

			color.x = sphereColor.x * multiplier;
			color.y = sphereColor.y * multiplier;
			color.z = sphereColor.z * multiplier;
		}

		return color;
	}
}

