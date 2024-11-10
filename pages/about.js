import React from "react"

import Layout from "../src/components/layout"
import Seo from "../src/components/seo"
import Image from "next/image"
import Heading from "../src/components/Heading"

const About = () => {
  return (
    <Layout>
      <Seo title="About" />
      <div className="container">
        <Heading level={1}>Hey, I'm Amit 👋</Heading>

        <div className="flex flex-col-reverse gap-8 items-center mb-10 md:flex-row">
          <div className="md:w-2/3">
            <p>
              I'm a Full Stack Engineer based in{" "}
              <a
                href="https://en.wikipedia.org/wiki/Mumbai"
                target="_blank"
                rel="noreferrer"
              >
                Mumbai
              </a>
              , India. I'm passionate about building scalable web applications
              using Python frameworks like FastAPI and Django for backend
              development. I also enjoy creating responsive and interactive user
              interfaces with React and Next.js. Recently, I've been exploring
              Java Spring Boot to expand my backend development skills.
            </p>

            <p>
              Outside of work I like to spend my time drinking coffee, working
              out, gaming, and watching Premier League.
            </p>
          </div>
          <div className="md:w-1/3">
            <Image
              src="/images/profile.jpg"
              alt="Profile Picture"
              layout="responsive"
              width={589}
              height={700}
            />
          </div>
        </div>

        <Heading level={2}>Tech Skills</Heading>
        <ul className="list-disc columns-3 ml-6">
          <li>Python</li>
          <li>JavaScript</li>
          <li>Java</li>
          <li>React</li>
          <li>Next.js</li>
          <li>Node.js</li>
          <li>FastAPI</li>
          <li>Django</li>
          <li>AWS</li>
          <li>GCP</li>
          <li>Linux</li>
          <li>XGBoost</li>
          <li>OpenCV</li>
          <li>Tableau</li>
          <li>MySQL</li>
          <li>MongoDB</li>
          <li>Git</li>
          <li>Selenium</li>
          <li>Docker</li>
          <li>KubeFlow</li>
        </ul>
      </div>
    </Layout>
  )
}

export default About
