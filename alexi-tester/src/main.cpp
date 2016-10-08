#include "ffn.hpp"
#include "fann_ffn.hpp"
#include "rnn.hpp"
#include "data_preprocessor.hpp"
#include "training_set.hpp"

#include <random>
#include <memory>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char* argv[]) {
    //Get options
    po::options_description desc("supported options");
    desc.add_options()
        ("testset", po::value<std::string>()->required())
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::vector<fs::path> paths;
    paths.push_back(fs::path(vm["testset"].as<std::string>()));

    data_preprocessor testset(paths);
    testset.run_processor(FILTER);
    //testset.run_processor(AVERAGE);
    testset.run_processor(LOWPASS);
    testset.run_processor(NORMALIZE);
    testset.run_processor(THRESHOLD);

    testset.write_csv("./test_dx.csv", 0);
    testset.write_csv("./test_dy.csv", 1);
    testset.write_csv("./test_dtheta.csv", 2);

    training_set tset_dx(testset.get_frames(), testset.get_images(), testset.get_diff_images(), DX);
    tset_dx.save_fann_data("./test_dx.data");
    training_set tset_dy(testset.get_frames(), testset.get_images(), testset.get_diff_images(), DY);
    tset_dy.save_fann_data("./test_dy.data");
    training_set tset_dtheta(testset.get_frames(), testset.get_images(), testset.get_diff_images(), DTHETA);
    tset_dtheta.save_fann_data("./test_dtheta.data");

    //std::shared_ptr<fann_fnn> dx_ann = std::make_shared<fann_ffn>("fann_dx.net");
    //std::shared_ptr<fann_fnn> dy_ann = std::make_shared<fann_ffn>("fann_dy.net");
    //std::shared_ptr<fann_fnn> dtheta_ann = std::make_shared<fann_ffn>("fann_dtheta.net");
    fann_ffn dx_ann("fann_dx.net");
    fann_ffn dy_ann("fann_dy.net");
    fann_ffn dtheta_ann("fann_dtheta.net");
    std::cout << "MSE dx: " << dx_ann.test("./test_dx.data") << std::endl;
    std::cout << "MSE dy: " << dy_ann.test("./test_dy.data") << std::endl;
    std::cout << "MSE dtheta: " << dtheta_ann.test("./test_dtheta.data") << std::endl;

    double stats[5][2];
    std::ifstream fin("./stats.dat");
    std::string line;
    int idx = 0;
    while(std::getline(fin, line)) {
        std::istringstream iss(line);
        iss >> stats[idx][0] >> stats[idx][1];
        idx++;
    }

    std::ofstream fout("./test_path.csv");

    for(auto frame : testset.get_frames()) {
        float in[4] = {(frame.left  - ((stats[0][0]+stats[0][1])/2.0))/((stats[0][0]-stats[0][1])/2.0)
                      ,(frame.right - ((stats[1][0]+stats[1][1])/2.0))/((stats[1][0]-stats[1][1])/2.0)
                      ,(frame.pitch - ((stats[2][0]+stats[2][1])/2.0))/((stats[2][0]-stats[2][1])/2.0)
                      ,(frame.roll  - ((stats[3][0]+stats[3][1])/2.0))/((stats[3][0]-stats[3][1])/2.0)};

        /*std::cout << frame.left << std::endl;
        float norm = (frame.left  - ((stats[0][0]+stats[0][1])/2.0))/((stats[0][0]-stats[0][1])/2.0);
        std::cout << norm << std::endl;
        std::cout << norm*((stats[0][0]-stats[0][1])/2.0) + ((stats[0][0]+stats[0][1])/2.0) << std::endl << std::endl;*/

        float* dx = dx_ann.predict(in);
        float* dy = dy_ann.predict(in);
        float* dtheta = dtheta_ann.predict(in);

        /*float dx_out = dx[0]*2.5;
        float dy_out = dy[0]*2.5;
        float dtheta_out = dtheta[0]*-3.0;
        if(frame.vdl == 0) {
            dx_out *= 0.25;
            dy_out *= 0.25;
        }

        float speed = sqrt(pow(dx_out, 2.0) + pow(dy_out, 2.0));
        if(frame.vdl < 0.0) {
            speed *= -1.0;
        }*/
        float dx_out = dx[0];
        float dy_out = dy[0];
        float dtheta_out = dtheta[0];
        float speed = sqrt(pow(dx_out, 2.0) + pow(dy_out, 2.0));

        fout << dx_out << "," << dy_out << "," << dtheta_out << "," << speed << std::endl;
    }

    fout.close();
}
