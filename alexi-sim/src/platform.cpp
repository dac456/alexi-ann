#include "platform.hpp"
#include <random>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.25, 1.0);

std::ofstream platform::_log = std::ofstream("./path.csv");

platform::platform(SDL_Surface* disp, std::map<std::string, std::shared_ptr<fann_ffn>> ann, fake_imu_ptr imu, double init_x, double init_y)
    : _ticks(0)
    , _rand(0.0)
    , _display(disp)
    , _ann(ann)
    , _imu(imu)
    , _pos_x(init_x)
    , _pos_y(init_y)
    , _yaw(0.0)
    , _left(0.0)
    , _last_dx(0.0)
    , _last_dy(0.0)
    , _last_dtheta(0.0)
    , _last_speed(0.0)
    , _right(0.0)
    , _num_inputs(50)
{
    _input_dx.resize(3, _num_inputs);
    _input_dx = 0.0;
    _input_dy.resize(3, _num_inputs);
    _input_dy = 0.0;
    _input_dtheta.resize(3, _num_inputs);
    _input_dtheta = 0.0;

    remove("./sim_path.csv");
}

void platform::step(double width, double height){
    const size_t interval = 2000;

    if(_ticks < interval){
        _desired_linear_velocity = _rand * 3.0f;
        _desired_angular_velocity = 0.0f;
    }
    else if(_ticks >= interval && _ticks < interval*2){
        _desired_linear_velocity = -_rand * 3.0f;
        _desired_angular_velocity = 0.0f;
    }
    else if(_ticks >= interval*2 && _ticks < interval*3){
        _desired_linear_velocity = 0.0f;
        _desired_angular_velocity = _rand * 1.57f;
    }
    else if(_ticks >= interval*3 && _ticks < interval*4){
        _desired_linear_velocity = 0.0f;
        _desired_angular_velocity = -_rand * 1.57f;
    }
    else if(_ticks >= interval*4 && _ticks < interval*5){
        _desired_linear_velocity = _rand * 3.0f;
        _desired_angular_velocity = _rand * 1.57f;
    }
    else if(_ticks >= interval*5 && _ticks < interval*6){
        _desired_linear_velocity = -_rand * 3.0f;
        _desired_angular_velocity = -_rand * 1.57f;
    }
    else if(_ticks >= interval*6 && _ticks < interval*7){
        _desired_linear_velocity = -_rand * 3.0f;
        _desired_angular_velocity = _rand * 1.57f;
    }
    else if(_ticks >= interval*7 && _ticks < interval*8){
        _desired_linear_velocity = _rand * 3.0f;
        _desired_angular_velocity = -_rand * 1.57f;
    }

    if(_ticks % 250 == 0){
        _rand = distribution(generator);
    }

    _ticks++;
    //_desired_linear_velocity = 0.0f;
    //_desired_angular_velocity = 1.8f;
    _move();

    std::cout << "pitch: " << _imu->get_accel_pitch() << std::endl;
    std::cout << "roll: " << _imu->get_accel_roll() << std::endl;
    std::cout << _left << " " << _right << std::endl;
    //float in[3] = {_left, _right, _imu->get_accel_pitch()};
    /*if(_num_inputs < 50){
        _num_inputs++;
        _input_dx.resize(4, _num_inputs);
        _input_dy.resize(4, _num_inputs);
        _input_dtheta.resize(4, _num_inputs);
    }*/

    //blaze::DynamicVector<double, blaze::columnVector> in_dx_current(4);
    float in_dx_current[5] = {_left, _right, _imu->get_accel_pitch(), _imu->get_accel_roll(), _last_dx};
    //in_dx_current = {_desired_linear_velocity, _desired_angular_velocity, _imu->get_accel_pitch(), _last_dx};

    //blaze::DynamicVector<double, blaze::columnVector> in_dy_current(4);
    float in_dy_current[5] = {_left, _right, _imu->get_accel_pitch(), _imu->get_accel_roll(), _last_dy};
    //in_dy_current = {_desired_linear_velocity, _desired_angular_velocity, _imu->get_accel_pitch(), _last_dy};

    //blaze::DynamicVector<double, blaze::columnVector> in_dtheta_current(4);
    float in_dtheta_current[5] = {_left, _right, _imu->get_accel_pitch(), _imu->get_accel_roll(), _last_dtheta};
    //in_dtheta_current = {_desired_linear_velocity, _desired_angular_velocity, _imu->get_accel_pitch(), _last_dtheta};

    double stats[10][2];
    std::ifstream fin("./stats.dat");
    std::string line;
    int idx = 0;
    while(std::getline(fin, line)) {
        std::istringstream iss(line);
        iss >> stats[idx][0] >> stats[idx][1];
        idx++;
    }

    //float in[3] = {(_left - stats[0][0])/stats[0][1], (_right - stats[1][0])/stats[1][1], (_imu->get_accel_pitch() - stats[2][0])/stats[2][1]/*, _imu->get_accel_roll()*/};
    float norm_left = (_left - ((stats[0][0]+stats[0][1])/2.0))/((stats[0][0]-stats[0][1])/2.0);
    float norm_right = (_right - ((stats[1][0]+stats[1][1])/2.0))/((stats[1][0]-stats[1][1])/2.0);
    float norm_pitch = (_imu->get_accel_pitch() - ((stats[2][0]+stats[2][1])/2.0))/((stats[2][0]-stats[2][1])/2.0);
    float norm_roll = (_imu->get_accel_roll() - ((stats[3][0]+stats[3][1])/2.0))/((stats[3][0]-stats[3][1])/2.0);
    float in[4] = {norm_left, norm_right, norm_pitch, norm_roll};

    for(size_t i = 0; i < _num_inputs-1; i++){
        column(_input_dx, i) = column(_input_dx, i+1);
        column(_input_dy, i) = column(_input_dy, i+1);
        column(_input_dtheta, i) = column(_input_dtheta, i+1);
    }
    //column(_input_dx, _num_inputs-1) = in/*_dx_current*/;
    //column(_input_dy, _num_inputs-1) = in/*_dy_current*/;
    //column(_input_dtheta, _num_inputs-1) = in/*_dtheta_current*/;



    float* /*blaze::DynamicVector<double, blaze::columnVector>*/ out_dx = _ann["dx"]->predict(in/*_input_dx*/);
    float* /*blaze::DynamicVector<double, blaze::columnVector>*/ out_dy = _ann["dy"]->predict(in/*_input_dy*/);
    float* /*blaze::DynamicVector<double, blaze::columnVector>*/ out_dtheta = _ann["dtheta"]->predict(in/*_input_dtheta*/);
    float* out_speed = _ann["speed"]->predict(in);

    /*if(out_dtheta[0] > 1.0 || out_dtheta[0] < 1.0) {
        //out_dtheta[0] = _last_dtheta;
        //out_dtheta[0] = 0.0;
    }
    if(out_speed[0] > 1.0 || out_speed[0] < 1.0) {
        //out_speed[0] = _last_speed;
        //out_speed[0] = 0.0;
    }*/

    std::ofstream fout("./sim_path.csv", std::ofstream::app);
    float sp = sqrt(pow(out_dx[0], 2.0) + pow(out_dy[0], 2.0));
    fout << out_dx[0] << "," << out_dy[0] << "," << out_dtheta[0] << "," << out_speed[0] << "," << norm_pitch  << "," << norm_left << "," << norm_right << std::endl;
    fout.close();

    out_dtheta[0] = (out_dtheta[0]*6.0)*((stats[4][0]-stats[4][1])/2.0) + ((stats[4][0]+stats[4][1])/2.0);
    out_dx[0] = (out_dx[0]*2.0)*((stats[5][0]-stats[5][1])/2.0) + ((stats[5][0]+stats[5][1])/2.0);
    out_dy[0] = (out_dy[0]*2.0)*((stats[6][0]-stats[6][1])/2.0) + ((stats[6][0]+stats[6][1])/2.0);
    out_speed[0] = (out_speed[0]*2.0)*((stats[9][0]-stats[9][1])/2.0) + ((stats[9][0]+stats[9][1])/2.0);
    if(_desired_linear_velocity < 0.0f) {
        //out_dtheta[0] *= -1.0;
    }

    if(_desired_linear_velocity == 0.0/* && _desired_angular_velocity == 0.0*/){
        //out_dx[0] *= 0.25;
        //out_dy[0] *= 0.25;
        //out_dtheta[0] = 0.0;
    }
    //if(_desired_angular_velocity == 0.0f){
    //    out_dtheta[0] = 0.0;
    //}
    _last_dx = out_dx[0];
    _last_dy = out_dy[0];
    _last_dtheta = out_dtheta[0];
    _last_speed = out_speed[0];

    //float speed = sqrt( pow(out_dx[0]*1.0, 2.0) + pow(out_dy[0]*1.0, 2.0) );
    float speed = out_speed[0];

    if(_desired_linear_velocity < 0.0f){
        speed *= -1.0f;
    }

    /*if(_desired_angular_velocity > 0){
        _yaw -= out_dtheta[0];
    }
    else{
        _yaw += out_dtheta[0];
    }*/
    _yaw += out_dtheta[0];

    _pos_x += speed*cos(_yaw);
    _pos_y += speed*sin(_yaw);
    //_pos_x += out_dx[0];
    //_pos_y += out_dy[0];

    _log << norm_left << "," << norm_right << "," << norm_pitch << "," << norm_roll << "," << out_dx[0]*2.5 << "," << out_dy[0]*2.5 << "," << out_dtheta[0]*-3.0 << std::endl;

    if(_pos_x > width * 0.5){
        _pos_x -= width;
    }
    if(_pos_x < -width * 0.5){
        _pos_x += width;
    }
    if(_pos_y > height * 0.5){
        _pos_y -= height;
    }
    if(_pos_y < -height * 0.5){
        _pos_y += height;
    }

    //_move();
    _last_input = in;
}

void platform::_move(){
    const double r = 0.5; //wheel radius (m)
    const double L = 3.2679; //wheel base (m)

    _left = (_desired_linear_velocity - (L*_desired_angular_velocity*0.5)) / r;
    _right = (_desired_linear_velocity + (L*_desired_angular_velocity*0.5)) / r;
}

double platform::get_pos_x(){
    return _pos_x;
}

double platform::get_pos_y(){
    return _pos_y;
}

double platform::get_yaw(){
    return _yaw;
}

bool platform::is_inclined(){
    if(fabs((_imu->get_accel_pitch()*180.0) / M_PI) > 20.0) return true;
    else return false;
}

float* platform::get_last_input(){
    return _last_input;
}
