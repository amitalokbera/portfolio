import React from "react"
import Layout from "../src/components/layout"
import Seo from "../src/components/seo"
import Image from "next/image"
import Heading from "../src/components/Heading"

const Uses = () => {
  return (
    <Layout>
        <Seo title="Uses" />
        <div className="container">

          <Heading level={1}>Uses</Heading>

          <div className="flex flex-col-reverse justify-between gap-8 md:flex-row">

              <div className="w-full md:w-1/2">
                  <Heading level={2}>Software &amp; Hardware I Use</Heading>

                  <Heading level={3}>Development</Heading>
                  <ul className="list-disc ml-5 mb-8">
                      <li>VS Code</li>
                      <li>Jupyter Notebook</li>
                      <li>Jupyter Lab</li>
                      <li>Colab</li>
                      <li>Docker</li>
                      <li>Microsoft Excel</li>
                      <li>Google Chrome</li>
                  </ul>

                  <Heading level={3}>Office Setup</Heading>
                  <ul className="list-disc ml-5 mb-8">
                      <li>Desktop Setup</li>
                        <ul>
                          <li>CPU - i7-7700K @4.8GHz</li>
                          <li>Motherboard - MSI Z270 Gaming Plus</li>
                          <li>RAM - 16GB</li>
                          <li>GPU - GTX 1050Ti</li>
                          <li>SSD - 512GB</li>
                        </ul>
                      <li>Laptop Setup</li>
                        <ul>
                          <li>MSI Modern 14 - Ryzen 5 4500U</li>
                        </ul>
                      <li>BenQ 24inch GW2280 - Primary Monitor</li>
                      <li>Samsung 24inch S22F350 - Secondary Monitor</li>
                      <li>RedGear Mechanical Keybaord</li>
                      <li>Logitech G402 Mouse</li>
                      <li>Oneplus Earbuds</li>
                      <li>Mi 720p Webcam</li>
                      <li>OneOdio Studio Headphones</li>
                  </ul>
              </div>
              <div className="w-full md:w-1/2">
                <Image src="/images/setup.jpg" alt="Office" layout="responsive" width={600} height={800} />
              </div>
          </div>
        </div>
    </Layout>
  )
}

export default Uses