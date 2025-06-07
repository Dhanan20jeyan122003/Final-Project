// app/page.tsx
import HeartForm from "./component/HeartForm";
import Navigation from "./component/Navigation";
import Header from "./component/Header";
import Portfolio from "./component/Portfolio";
import Services from "./component/Service";
import About from "./component/About";
import Heart from "./component/Heart";
import Footer from "./component/footer";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-pink-100 to-red-100 p-6">
        <Navigation/>
        <Header/>
        <About/>
       <Portfolio/>
       <Services/> 
       <HeartForm />  
       {/*  <Heart/>   */}
        <Footer/>
    </div>
  );
}
