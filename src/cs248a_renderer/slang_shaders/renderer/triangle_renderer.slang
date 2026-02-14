implementing "../renderer.slang";


float3 simpleShade(float3 rayDir, float3 normal, float3 albedo)
{
    float NdotL = max(dot(normalize(normal), normalize(-rayDir)), 0.0);
    return albedo * NdotL;
}

public struct RayMeshSampleResult
{
    public float t;
    public float3 normal;
    public float4 color;
}

public struct RayMeshIntersectionResult
{
    public RayHitResult closestHit;
    public Triangle hitTriangle;
}

public Optional<RayMeshSampleResult> meshSample(float2 uv, RendererUniform uniforms)
{
    // We generate a camera ray for the given uv coordinate.
    // You need to implement the `generateRay` function in `Camera` struct in `src/cs248a_renderer/slang_shaders/model/camera.slang`
    Ray ray = uniforms.camera.generateRay(uv);
    RayHitResult result;

    Optional<RayMeshSampleResult> allResult;
    // Ray mesh intersection test.
    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult.hasValue)
    {
        result = rayMeshIntersectionResult.value.closestHit;
        float3 hitPoint = ray.origin + result.t * ray.direction;
        float3 barycentricCoord = rayMeshIntersectionResult.value.hitTriangle.calculateBarycentricCoord(hitPoint);
        float3 normal = result.normal;
        float2 tex_uv = rayMeshIntersectionResult.value.hitTriangle.getUV(barycentricCoord);
        float4 color = shadeTriangle(
            uv, 
            uniforms, 
            rayMeshIntersectionResult.value.hitTriangle, 
            ray, 
            tex_uv, 
            rayMeshIntersectionResult.value.hitTriangle.getNormal(barycentricCoord), 
            barycentricCoord
        );
        allResult = Optional<RayMeshSampleResult>(RayMeshSampleResult(result.t, normal, color));
    }

    return allResult;
}

public Optional<RayMeshIntersectionResult> rayMeshIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<RayHitResult> result = Optional<RayHitResult>();

    // IMPORTANT CHANGE: We need to store the hit triangle.
    Optional<RayMeshIntersectionResult> allResult = Optional<RayMeshIntersectionResult>();
    Optional<Triangle> hitTriangle = Optional<Triangle>();
    // IMPORTANT CHANGE: We need to store the hit triangle.

    // TODO: Student implementation starts here.
    Optional<RayHitResult> closestHitResult;
    Optional<RayHitResult> hitResult;
    if (uniforms.useBVH) {
        Optional<BVH<Triangle>.HitResult> bvhHitResult = uniforms.bvh.hit(ray);
        if (bvhHitResult.hasValue) {
            closestHitResult = Optional<RayHitResult>(bvhHitResult.value.rayHitResult);
            hitTriangle = Optional<Triangle>(bvhHitResult.value.primitive);
        }
    } else {
        for (uint i = 0; i < uniforms.triangleCount; i++)
        {
            Triangle triangle = uniforms.triangleBuf[i];
            hitResult = triangle.hit(ray);
            if (!hitResult.hasValue) {}
            else if (!closestHitResult.hasValue) {
                closestHitResult = hitResult;
                hitTriangle = Optional<Triangle>(triangle);
            }
            else if (hitResult.value.t < closestHitResult.value.t) {
                closestHitResult = hitResult;
                hitTriangle = Optional<Triangle>(triangle);
            }  
        }
    }
    result = closestHitResult;

    
    // TODO: Student implementation ends here.

    if (result.hasValue && hitTriangle.hasValue)
    {
        allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle.value));
    }

    return allResult;
}

// Helper function to get texture UV at an offset screen position
// Returns an Optional - empty if the offset ray doesn't hit the triangle
Optional<float2> getTexUVAtOffset(RendererUniform uniforms, Triangle triangle, float2 offsetUV)
{
    Ray ray = uniforms.camera.generateRay(offsetUV);
    Optional<RayHitResult> result = triangle.hit(ray);
    if (!result.hasValue) {
        return Optional<float2>();
    }
    float3 hitPoint = ray.origin + result.value.t * ray.direction;
    float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
    return Optional<float2>(triangle.getUV(barycentricCoord));
}

// Compute the level of the texture at the hit point.
// @param uv: The uv coordinates of the hit point.
// @param uniforms: The renderer uniforms.
// @param triangle: The triangle that was hit.
// @param tex_uv: The uv coordinates of the texture at the hit point.
// @return: The level of the texture at the hit point.
public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.

    // One pixel step in normalized screen coordinates
    float dx = 1.0 / float(uniforms.camera.canvasSize.x);
    float dy = 1.0 / float(uniforms.camera.canvasSize.y);

    // Get texture UVs at screen positions offset by one pixel in x and y
    // Optional<float2> texUV_opt = getTexUVAtOffset(uniforms, triangle, uv);
    // if (!texUV_opt.hasValue) {
    //     return 0.0;
    // }
    Optional<float2> texUV_pos_dx_opt = getTexUVAtOffset(uniforms, triangle, uv + float2(dx, 0.0));
    Optional<float2> texUV_neg_dx_opt = getTexUVAtOffset(uniforms, triangle, uv - float2(dx, 0.0));
    Optional<float2> texUV_pos_dy_opt = getTexUVAtOffset(uniforms, triangle, uv + float2(0.0, dy));
    Optional<float2> texUV_neg_dy_opt = getTexUVAtOffset(uniforms, triangle, uv - float2(0.0, dy));

    // If offset rays miss the triangle, use the current tex_uv as fallback
   
    float2 texUV_dx = texUV_pos_dx_opt.hasValue ? texUV_pos_dx_opt.value : texUV_neg_dx_opt.hasValue ? texUV_neg_dx_opt.value : tex_uv;
    float2 texUV_dy = texUV_pos_dy_opt.hasValue ? texUV_pos_dy_opt.value : texUV_neg_dy_opt.hasValue ? texUV_neg_dy_opt.value : tex_uv;

    float2 duvdx = (texUV_dx - tex_uv) / dx;
    float2 duvdy = (texUV_dy - tex_uv) / dy;

    float rho_x = length(duvdx);
    float rho_y = length(duvdy);

    float rho = max(rho_x, rho_y);
    level = log2(max(rho, 1.0));

    return level;

    // TODO: Student implementation ends here.
}  

public float4 shadeTriangle(float2 uv, RendererUniform uniforms, Triangle triangle, Ray ray, float2 tex_uv, float3 normal, float3 barycentricCoord)
{
    if (uniforms.visualizeBarycentricCoords)
    {
        return float4(barycentricCoord, 1.0);
    }
    if (uniforms.visualizeTexUV)
    {
        return float4(tex_uv, 0.0, 1.0);
    }
    // Get the level of detail of the texture at the hit point.
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    // visualize level
    // if (uniforms.visualizeLevelOfDetail)
    // {
    //     uint discreteLevel = uint(level);
    //     return float4(float3(discreteLevel) / 4.0, 1.0);
    // }
    uint discreteLevel = uint(level);

    // Sample albedo from the material at the hit point.
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    float3 albedo = parameters.albedo;

    if (uniforms.visualizeLevelOfDetail || uniforms.visualizeAlbedo)
    {
        return float4(float3(albedo), 1.0);
    }

    // Shade the triangle.
    float3 color = simpleShade(ray.direction, normal, albedo);
    return float4(color, 1.0);
}
