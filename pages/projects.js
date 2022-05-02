import React from "react"
import Heading from "../src/components/Heading"
import Layout from "../src/components/layout"
import Seo from "../src/components/seo"

const Projects = () => {
    return (
        <Layout>
            <Seo title="Projects" />
            
            <div className="container">
              <Heading level={1} style={{marginBottom: "40px"}}>Projects I've Worked On</Heading>

              <div class="flex flex-col gap-10">
                
                <Heading level={2} style={{marginBottom: "0px"}}>Side Projects</Heading>
                
                <div className="flex flex-col md:grid md:grid-cols-2 gap-8 mb-12 md:mb-20">

                    <div className="rounded shadow-gray-400 shadow-lg p-6 border border-gray-100">
                        <Heading level={3} style={{marginBottom: "0px"}}>Microsoft Malware Detection</Heading>
                        <p className="text-sm">Python - Scikit-Learn - XGBoost</p>
                        <div class="flex justify-between items-center gap-4 mt-4">
                        </div>
                    </div>

                    <div className="rounded shadow-gray-400 shadow-lg p-6 border border-gray-100">
                        <Heading level={3} style={{marginBottom: "0px"}}>Pneumonia Detection onn CT Scan</Heading>
                        <p className="text-sm">Python - Scikit-Learn - TensorFlow 2 (Transfer Learning) - Flask API</p>
                        <div class="flex justify-between items-center gap-4 mt-4">
                        </div>
                    </div>

                    <div className="rounded shadow-gray-400 shadow-lg p-6 border border-gray-100">
                        <Heading level={3} style={{marginBottom: "0px"}}>Sentiment Analysis on Movie Review Prediction</Heading>
                        <p className="text-sm">Python - Scikit-Learn - Naive Bayes</p>
                        <div class="flex justify-between items-center gap-4 mt-4">
                        </div>
                    </div>

                    <div className="rounded shadow-gray-400 shadow-lg p-6 border border-gray-100">
                        <Heading level={3} style={{marginBottom: "0px"}}>BERT Classification Model</Heading>
                        <p className="text-sm">Python - TensorFlow 2 - BERT - LSTM</p>
                        <div class="flex justify-between items-center gap-4 mt-4">
                        </div>
                    </div>

                    <div className="rounded shadow-gray-400 shadow-lg p-6 border border-gray-100">
                        <Heading level={3} style={{marginBottom: "0px"}}>eCommerce Web Automation Tool</Heading>
                        <p className="text-sm">Python - Requests - PyQT6 - Selenium</p>
                        <div class="flex justify-between items-center gap-4 mt-4">
                        </div>
                    </div>

                </div>
                
                <Heading level={2} style={{marginBottom: "0px"}}>Awards</Heading>

                <div className="flex-col flex gap-8 mb-12 md:mb-20 md:flex-row">
                      <div className="w-full md:w-1/3">
                          <Heading level={3} style={{marginBottom: "0px"}}>MLRW 2022 : AI Driven Biomedical Hackathon</Heading>
                          <p className="text-xl mb-0">Kaggle competition</p>
                          <p className="text-sm">Python - Scikit-Learn - XGBoost - Decision Tree - TensorFlow 2</p>
                          <p className="text-xs mb-0">Hosted by <a href="https://www.elucidata.io/" target="_blank" className="text-indigo-700 hover:text-violet-500">Elucidata</a></p>
                      </div>
                      <div className="w-full md:w-2/3">
                          <p>Secured 3rd Place in Biomedical based NLP Competition</p>
                          <p>The problem is a text classification problem where you have to predict the target class ctrl. The text present in the various columns of training data is metadata associated with omics data. It is used to get information about the samples which were studied in any experiment for example, sex, age, organism, etc. From this text you need to predict whether some sample is a control or perturbation sample.</p>
                      </div>
                  </div>

                  <div className="flex-col flex gap-8 mb-12 md:mb-20 md:flex-row">
                    <div className="w-full md:w-1/3">
                        <Heading level={3} style={{marginBottom: "0px"}}>UltraMNIST</Heading>
                        <p className="text-xl mb-0">Kaggle Competition</p>
                        <p className="text-sm">Python - Scikit-Learn - TensorFlow  2 - OpenCV - Pillow</p>
                        <p className="text-xs mb-0">Hosted by <a href="https://nvidia.com" target="_blank" className="text-indigo-700 hover:text-violet-500">Nvidia</a></p>
                    </div>
                    <div className="w-full md:w-2/3">
                        <p>Secured 7th position in Computer Vision-based Kaggle Competition </p>
                        <p>We pose the problem of the classification of UltraMNIST digits. UltraMNIST dataset comprises very large-scale images, each of 4000x4000 pixels with 3-5 digits per image. Each of these digits has been extracted from the original MNIST dataset. Your task is to predict the sum of the digits per image, and this number can be anything from 0 to 27.</p>
                    </div>
                </div>

                

              </div>
          </div>

        </Layout>
    )
}

export default Projects