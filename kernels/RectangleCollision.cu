__global__ void check_collisions(
    float x1_robot, float y1_robot, float x2_robot, float y2_robot,
    float *x1_obs, float *y1_obs, float *x2_obs, float *y2_obs,
    bool *collisions, int *indexes)
{
    int obstacleId = threadIdx.x;
    
    bool xcol = ((x1_obs[obstacleId] <= x1_robot && x1_robot <= x2_obs[obstacleId]) 
            || (x1_obs[obstacleId] <= x2_robot && x2_robot <= x2_obs[obstacleId])) 
            || ( x1_robot <= x1_obs[obstacleId] && x2_robot >= x2_obs[obstacleId]);

    bool ycol = ((y1_obs[obstacleId] <= y1_robot && y1_robot <= y2_obs[obstacleId]) 
            || (y1_obs[obstacleId] <= y2_robot && y2_robot <= y2_obs[obstacleId])) 
            || ( y1_robot <= y1_obs[obstacleId] && y2_robot >= y2_obs[obstacleId]);

    collisions[obstacleId] = (xcol && ycol);
}